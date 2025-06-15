import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
import os

# 导入我们定义的模块
from hetero_graph_builder import HeteroGraphBuilder
from hetero_gnn_model import HeteroGNNModel, DualTowerPredictor

logger = logging.getLogger(__name__)


class ZILNLoss(nn.Module):
    """零膨胀对数正态分布损失"""
    
    def forward(self, p_zero, mu, sigma, treatment_L):
        """
        计算ZILN密度损失
        Args:
            p_zero: 零概率预测 [B, 1]
            mu, sigma: 对数正态分布参数 [B]
            treatment_L: 实际处理值 [B]
        """
        eps = 1e-8
        
        # 分离零值和非零值
        is_zero = (treatment_L == 0).float()
        is_positive = 1 - is_zero
        
        # 零部分的损失
        loss_zero = -is_zero * torch.log(1 - p_zero.squeeze() + eps)
        
        # 正值部分的损失
        log_L = torch.log(treatment_L + eps)
        normal_pdf = -0.5 * ((log_L - mu) / sigma) ** 2 - torch.log(sigma + eps)
        loss_positive = -is_positive * (
            torch.log(p_zero.squeeze() + eps) + 
            normal_pdf - torch.log(treatment_L + eps)
        )
        
        return (loss_zero + loss_positive).mean()


class CausalLoss(nn.Module):
    """完整的三头联合损失函数"""
    
    def __init__(self, beta: float = 1.0, lambda_dsl: float = 0.1):
        super().__init__()
        self.beta = beta
        self.lambda_dsl = lambda_dsl
        self.ziln_loss = ZILNLoss()
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, predictions: Dict, targets: Dict, dsl_scores: torch.Tensor):
        """
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            dsl_scores: DSL分数
        """
        # 1. 密度估计损失 (Treatment Tower)
        L_den = self.ziln_loss(
            predictions['p_zero'],
            predictions['mu'],
            predictions['sigma'],
            targets['treatment_L']
        )
        
        # 2. IPW加权的结果预测损失 (Outcome Tower)
        # 计算IPW权重
        with torch.no_grad():
            # 简化的密度计算
            is_zero = (targets['treatment_L'] == 0).float()
            density = is_zero * (1 - predictions['p_zero'].squeeze()) + \
                     (1 - is_zero) * predictions['p_zero'].squeeze()
            ipw_weights = 1.0 / (density + 1e-8)
            ipw_weights = torch.clamp(ipw_weights, 0.1, 10.0)  # 裁剪极值
        
        # BCE损失
        bce_raw = self.bce_loss(predictions['outcome'].squeeze(), targets['outcome'])
        L_out = (ipw_weights * bce_raw).mean()
        
        # 3. DSL结构惩罚
        L_dsl = dsl_scores.mean()
        
        # 总损失
        total_loss = L_den + self.beta * L_out + self.lambda_dsl * L_dsl
        
        return {
            'total': total_loss,
            'density': L_den,
            'outcome': L_out,
            'dsl': L_dsl
        }


class CausalGraphDataset:
    """因果图数据集"""
    
    def __init__(self, 
                 user_features: np.ndarray,
                 user_edges: np.ndarray,
                 treatments: np.ndarray,
                 outcomes: np.ndarray,
                 graph_builder: HeteroGraphBuilder):
        
        self.user_features = user_features
        self.user_edges = user_edges
        self.treatments = treatments
        self.outcomes = outcomes
        self.graph_builder = graph_builder
        
        # 构建完整图
        self.full_graph = graph_builder.build_graph(
            user_features, user_edges, outcomes
        )
    
    def get_batch_subgraph(self, batch_idx: torch.Tensor):
        """获取批次子图"""
        # 这里简化处理，实际应该使用更高效的子图采样
        return self.full_graph, batch_idx


class Trainer:
    """训练器"""
    
    def __init__(self,
                 model: HeteroGNNModel,
                 predictor: DualTowerPredictor,
                 dataset: CausalGraphDataset,
                 config: Dict):
        
        self.model = model
        self.predictor = predictor
        self.dataset = dataset
        self.config = config
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # 损失函数
        self.criterion = CausalLoss(
            beta=config.get('beta', 1.0),
            lambda_dsl=config.get('lambda_dsl', 0.1)
        )
        
        # 初始化wandb
        if config.get('use_wandb', False):
            wandb.init(project="causal-graph-uplift", config=config)
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        self.predictor.train()
        
        n_samples = len(self.dataset.treatments)
        batch_size = self.config['batch_size']
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        epoch_losses = {'total': 0, 'density': 0, 'outcome': 0, 'dsl': 0}
        
        # 随机打乱
        perm = torch.randperm(n_samples)
        
        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch}')
        for i in pbar:
            # 获取批次索引
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_idx = perm[start_idx:end_idx]
            
            # 获取子图（这里简化处理）
            graph, node_idx = self.dataset.get_batch_subgraph(batch_idx)
            
            # 前向传播
            gnn_outputs = self.model(graph)
            
            # 获取批次的用户嵌入
            user_embed = gnn_outputs['user_embed'][node_idx]
            dsl_embed = gnn_outputs['dsl_embed'][node_idx]
            
            # 获取批次标签
            batch_treatments = torch.FloatTensor(
                self.dataset.treatments[batch_idx]
            )
            batch_outcomes = torch.FloatTensor(
                self.dataset.outcomes[batch_idx]
            )
            
            # 双塔预测
            predictions = self.predictor(user_embed, batch_treatments)
            
            # 计算DSL分数（简化版）
            dsl_scores = dsl_embed.abs().mean(dim=1)
            
            # 计算损失
            targets = {
                'treatment_L': batch_treatments,
                'outcome': batch_outcomes
            }
            losses = self.criterion(predictions, targets, dsl_scores)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.predictor.parameters()),
                self.config.get('clip_grad', 1.0)
            )
            
            self.optimizer.step()
            
            # 更新统计
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'den': losses['density'].item(),
                'out': losses['outcome'].item()
            })
        
        # 学习率调度
        self.scheduler.step()
        
        # 返回平均损失
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        return epoch_losses
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        self.predictor.eval()
        
        with torch.no_grad():
            # 在整个图上进行推理
            gnn_outputs = self.model(self.dataset.full_graph)
            
            # 生成剂量-响应曲线
            n_users = gnn_outputs['user_embed'].shape[0]
            treatment_grid = torch.linspace(0, 50000, 50)
            
            response_curves = []
            for L in treatment_grid:
                L_batch = torch.full((n_users,), L)
                predictions = self.predictor(gnn_outputs['user_embed'], L_batch)
                response_curves.append(predictions['outcome'].mean().item())
            
            # 计算AUUC等指标
            # 这里简化处理，实际应该实现完整的评估逻辑
            auuc = np.trapz(response_curves, treatment_grid) / 50000
        
        return {
            'auuc': auuc,
            'response_curve': response_curves
        }
    
    def train(self):
        """完整训练流程"""
        logger.info("开始训练...")
        
        best_auuc = -float('inf')
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 评估
            if epoch % self.config.get('eval_interval', 5) == 0:
                eval_metrics = self.evaluate()
                
                logger.info(
                    f"Epoch {epoch}: "
                    f"Loss={train_losses['total']:.4f}, "
                    f"AUUC={eval_metrics['auuc']:.4f}"
                )
                
                # 保存最佳模型
                if eval_metrics['auuc'] > best_auuc:
                    best_auuc = eval_metrics['auuc']
                    self.save_checkpoint(epoch, eval_metrics)
                
                # 记录到wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'epoch': epoch,
                        'train/total_loss': train_losses['total'],
                        'train/density_loss': train_losses['density'],
                        'train/outcome_loss': train_losses['outcome'],
                        'train/dsl_loss': train_losses['dsl'],
                        'eval/auuc': eval_metrics['auuc'],
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
        
        logger.info(f"训练完成！最佳AUUC: {best_auuc:.4f}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'predictor_state': self.predictor.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = os.path.join(
            self.config.get('save_dir', './checkpoints'),
            f'model_epoch{epoch}_auuc{metrics["auuc"]:.4f}.pt'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存到 {save_path}")


def main():
    """主函数"""
    # 配置
    config = {
        'lr': 1e-3,
        'batch_size': 256,
        'epochs': 100,
        'eval_interval': 5,
        'beta': 1.0,
        'lambda_dsl': 0.1,
        'clip_grad': 1.0,
        'weight_decay': 1e-5,
        'use_wandb': False,
        'save_dir': './checkpoints'
    }
    
    # 生成模拟数据
    n_users = 5000
    n_features = 30
    
    # 用户特征（20%缺失）
    user_features = np.random.randn(n_users, n_features)
    missing_mask = np.random.rand(n_users, n_features) < 0.2
    user_features[missing_mask] = np.nan
    
    # 用户社交网络
    n_edges = 15000
    user_edges = np.random.randint(0, n_users, size=(2, n_edges))
    
    # 处理变量（额度）
    treatments = np.random.lognormal(10, 1.5, n_users)
    treatments[np.random.rand(n_users) < 0.3] = 0  # 30%零值
    
    # 结果变量（逾期）
    # 简单的模拟：逾期概率与额度正相关
    base_prob = 0.05
    treatment_effect = treatments / 100000
    outcomes = (np.random.rand(n_users) < (base_prob + treatment_effect)).astype(float)
    
    # 构建图
    logger.info("构建异质图...")
    graph_builder = HeteroGraphBuilder(lambda_missing=0.1)
    
    # 创建数据集
    dataset = CausalGraphDataset(
        user_features, user_edges, treatments, outcomes, graph_builder
    )
    
    # 创建模型
    logger.info("初始化模型...")
    model = HeteroGNNModel(
        user_in_dim=n_features,
        feat_embed_dim=32,
        hidden_dim=128,
        out_dim=64,
        num_heads=4,
        num_cau_layers=2
    )
    
    predictor = DualTowerPredictor(embed_dim=64, hidden_dim=128)
    
    # 创建训练器
    trainer = Trainer(model, predictor, dataset, config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()