import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DMLCATEHead(nn.Module):
    """双重机器学习(DML)模块，用于估计单个特征的CATE"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 第一阶段：估计Y和T的残差
        self.outcome_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.treatment_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, X_minus_j: torch.Tensor, T: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X_minus_j: 除了第j个特征外的所有特征 [B, F-1]
            T: 处理变量（第j个特征） [B, 1]
            Y: 软标签 [B, 1]
        Returns:
            theta_j: 该特征的CATE估计值 [B, 1]
        """
        # 第一阶段：计算残差
        Y_pred = self.outcome_model(X_minus_j)
        T_pred = self.treatment_model(X_minus_j)
        
        Y_residual = Y - Y_pred
        T_residual = T - T_pred
        
        # 第二阶段：线性回归估计CATE
        # theta = (T_residual^T * T_residual)^(-1) * T_residual^T * Y_residual
        numerator = (T_residual * Y_residual).sum()
        denominator = (T_residual * T_residual).sum() + 1e-8
        theta_j = numerator / denominator
        
        return theta_j.unsqueeze(0)


class CATEEstimator:
    """多头DML-CATE估计器"""
    
    def __init__(self, n_features: int, hidden_dim: int = 64, n_folds: int = 5):
        self.n_features = n_features
        self.n_folds = n_folds
        self.heads = nn.ModuleList([
            DMLCATEHead(n_features - 1, hidden_dim) for _ in range(n_features)
        ])
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=1e-3)
    
    def fit(self, X: np.ndarray, Y_soft: np.ndarray, epochs: int = 100):
        """
        训练所有特征的CATE估计器
        Args:
            X: 原始特征矩阵 [N, F]
            Y_soft: 软标签 [N, 1]
        Returns:
            alpha: 每个特征的因果权重 [F]
        """
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y_soft).unsqueeze(1)
        
        n_samples = X.shape[0]
        feature_cates = []
        
        for j in range(self.n_features):
            logger.info(f"训练特征 {j} 的CATE估计器...")
            
            # 准备数据：X_minus_j 和 T_j
            mask = torch.ones(self.n_features, dtype=torch.bool)
            mask[j] = False
            X_minus_j = X[:, mask]
            T_j = X[:, j:j+1]
            
            # K折交叉验证
            fold_cates = []
            fold_size = n_samples // self.n_folds
            
            for fold in range(self.n_folds):
                # 划分训练集和验证集
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < self.n_folds - 1 else n_samples
                
                train_mask = torch.ones(n_samples, dtype=torch.bool)
                train_mask[val_start:val_end] = False
                val_mask = ~train_mask
                
                # 训练当前折
                for epoch in range(epochs):
                    self.optimizer.zero_grad()
                    
                    # 只在训练集上计算损失
                    theta = self.heads[j](X_minus_j[train_mask], T_j[train_mask], Y[train_mask])
                    
                    # 简单的MSE损失作为正则化
                    loss = F.mse_loss(theta, torch.zeros_like(theta))
                    loss.backward()
                    self.optimizer.step()
                
                # 在验证集上估计CATE
                with torch.no_grad():
                    val_theta = self.heads[j](X_minus_j[val_mask], T_j[val_mask], Y[val_mask])
                    fold_cates.append(val_theta.item())
            
            # 取所有折的平均值作为该特征的CATE
            feature_cates.append(np.mean(fold_cates))
        
        alpha = torch.tensor(feature_cates, dtype=torch.float32)
        alpha.requires_grad_(False)  # 冻结权重
        
        return alpha


class BayesianNetworkLearner:
    """贝叶斯网络结构学习器"""
    
    def __init__(self, scoring_method='bic'):
        self.scoring_method = scoring_method
        self.adjacency_matrix = None
    
    def fit(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用Hill-Climbing算法学习特征间的依赖结构
        Args:
            X: 特征数据框 [N, F]
        Returns:
            A_dag: 有向邻接矩阵 [F, F]
        """
        try:
            from pgmpy.estimators import HillClimbSearch, BicScore
            from pgmpy.models import BayesianNetwork
            
            # 执行结构学习
            hc = HillClimbSearch(X)
            if self.scoring_method == 'bic':
                scoring = BicScore(X)
            else:
                raise ValueError(f"不支持的评分方法: {self.scoring_method}")
            
            best_model = hc.estimate(scoring_method=scoring)
            
            # 构建邻接矩阵
            n_features = X.shape[1]
            A_dag = np.zeros((n_features, n_features))
            
            for edge in best_model.edges():
                i = X.columns.get_loc(edge[0])
                j = X.columns.get_loc(edge[1])
                A_dag[i, j] = 1
            
            self.adjacency_matrix = A_dag
            return A_dag
            
        except ImportError:
            logger.warning("pgmpy未安装，使用随机稀疏矩阵代替")
            # 生成随机稀疏DAG作为示例
            n_features = X.shape[1]
            A_dag = np.random.binomial(1, 0.1, size=(n_features, n_features))
            A_dag = np.tril(A_dag, -1)  # 确保是DAG
            return A_dag


class HeteroGraphBuilder:
    """异质图构建器"""
    
    def __init__(self, lambda_missing: float = 0.1):
        """
        Args:
            lambda_missing: 缺失值的门控权重
        """
        self.lambda_missing = lambda_missing
        self.cate_estimator = None
        self.bn_learner = BayesianNetworkLearner()
        self.feature_scaler = StandardScaler()
        self.alpha = None  # 特征因果权重
        self.A_feat = None  # 特征邻接矩阵
    
    def build_graph(self, 
                   user_features: np.ndarray,
                   user_edges: np.ndarray,
                   y_soft: Optional[np.ndarray] = None,
                   feature_names: Optional[List[str]] = None) -> HeteroData:
        """
        构建完整的异质图
        Args:
            user_features: 用户特征矩阵 [N, F]，可能包含缺失值
            user_edges: 用户社交边 [2, E_u]
            y_soft: 软标签，如果为None则随机生成
            feature_names: 特征名称列表
        Returns:
            graph: PyG异质图对象
        """
        n_users, n_features = user_features.shape
        
        # 1. 处理缺失值
        missing_mask = np.isnan(user_features)
        user_features_filled = np.nan_to_num(user_features, nan=0.0)
        
        # 2. 特征标准化
        user_features_scaled = self.feature_scaler.fit_transform(user_features_filled)
        
        # 3. 如果没有提供软标签，生成模拟的软标签
        if y_soft is None:
            logger.info("未提供软标签，使用模拟数据")
            y_soft = np.random.rand(n_users)
        
        # 4. 估计特征因果权重
        logger.info("开始估计特征因果权重...")
        self.cate_estimator = CATEEstimator(n_features)
        self.alpha = self.cate_estimator.fit(user_features_filled, y_soft)
        logger.info(f"特征因果权重: {self.alpha.numpy()}")
        
        # 5. 学习特征依赖结构
        logger.info("开始学习特征依赖结构...")
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(n_features)]
        df = pd.DataFrame(user_features_filled, columns=feature_names)
        self.A_feat = self.bn_learner.fit(df)
        
        # 对称化处理（用于无向图）
        A_feat_sym = ((self.A_feat + self.A_feat.T) > 0).astype(np.float32)
        feat_edges = np.where(A_feat_sym)
        
        # 6. 构建PyG异质图
        graph = HeteroData()
        
        # 添加用户节点
        graph['user'].x = torch.FloatTensor(user_features_scaled)
        graph['user'].num_nodes = n_users
        
        # 添加特征节点
        graph['feature'].alpha = self.alpha  # 因果权重
        graph['feature'].embed = nn.Parameter(torch.randn(n_features, 32))  # 可学习嵌入
        graph['feature'].num_nodes = n_features
        
        # 添加用户-用户边
        graph['user', 'social', 'user'].edge_index = torch.LongTensor(user_edges)
        
        # 添加特征-特征边
        if len(feat_edges[0]) > 0:
            graph['feature', 'rel', 'feature'].edge_index = torch.LongTensor(
                np.vstack(feat_edges)
            )
            # 边权重：保留原始方向信息
            edge_weights = []
            for i, j in zip(feat_edges[0], feat_edges[1]):
                if self.A_feat[i, j] > 0 and self.A_feat[j, i] > 0:
                    edge_weights.append(0.0)  # 双向
                elif self.A_feat[i, j] > 0:
                    edge_weights.append(1.0)  # 正向
                else:
                    edge_weights.append(-1.0)  # 反向
            graph['feature', 'rel', 'feature'].edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)
        
        # 7. 构建用户-特征二分边
        user_feat_edges = []
        edge_attrs = []
        
        for u in range(n_users):
            for f in range(n_features):
                user_feat_edges.append([u, f])
                
                # 计算边权重 w_{u,j} = (1 - delta_{u,j}) * alpha_j + delta_{u,j} * lambda
                delta_uj = float(missing_mask[u, f])
                w_uj = (1 - delta_uj) * self.alpha[f].item() + delta_uj * self.lambda_missing
                
                # 标准化特征值
                z_uj = user_features_scaled[u, f] if not missing_mask[u, f] else 0.0
                
                # 二维边属性：[w_uj, z_uj]
                edge_attrs.append([w_uj, z_uj])
        
        user_feat_edges = torch.LongTensor(user_feat_edges).T
        edge_attrs = torch.FloatTensor(edge_attrs)
        
        graph['user', 'has', 'feature'].edge_index = user_feat_edges
        graph['user', 'has', 'feature'].edge_attr = edge_attrs
        
        # 8. 保存元数据
        graph.metadata = {
            'created_at': datetime.now().isoformat(),
            'n_users': n_users,
            'n_features': n_features,
            'missing_rate': missing_mask.mean(),
            'alpha_version': f'v{datetime.now().strftime("%Y%m%d")}',
            'lambda_missing': self.lambda_missing
        }
        
        logger.info(f"异质图构建完成: {n_users}个用户, {n_features}个特征")
        logger.info(f"缺失率: {missing_mask.mean():.2%}")
        
        return graph
    
    def save_components(self, save_dir: str = './checkpoints'):
        """保存模型组件"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存因果权重
        torch.save(self.alpha, f'{save_dir}/alpha_v{datetime.now().strftime("%Y%m%d")}.pt')
        
        # 保存特征邻接矩阵
        np.save(f'{save_dir}/feature_adjacency.npy', self.A_feat)
        
        # 保存标准化器
        import joblib
        joblib.dump(self.feature_scaler, f'{save_dir}/feature_scaler.pkl')
        
        logger.info(f"模型组件已保存到 {save_dir}")


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    n_users = 1000
    n_features = 20
    
    # 用户特征（包含20%缺失值）
    user_features = np.random.randn(n_users, n_features)
    mask = np.random.rand(n_users, n_features) < 0.2
    user_features[mask] = np.nan
    
    # 用户社交关系（随机生成）
    n_edges = 3000
    user_edges = np.random.randint(0, n_users, size=(2, n_edges))
    
    # 构建异质图
    builder = HeteroGraphBuilder(lambda_missing=0.1)
    graph = builder.build_graph(user_features, user_edges)
    
    # 打印图的统计信息
    print("\n图统计信息:")
    print(f"用户节点数: {graph['user'].num_nodes}")
    print(f"特征节点数: {graph['feature'].num_nodes}")
    print(f"用户-用户边数: {graph['user', 'social', 'user'].edge_index.shape[1]}")
    print(f"用户-特征边数: {graph['user', 'has', 'feature'].edge_index.shape[1]}")
    if hasattr(graph['feature', 'rel', 'feature'], 'edge_index'):
        print(f"特征-特征边数: {graph['feature', 'rel', 'feature'].edge_index.shape[1]}")
    
    # 保存组件
    builder.save_components()