import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv, Linear
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
import math
from typing import Dict, Optional, Tuple


class RelationalGATConv(MessagePassing):
    """关系图注意力卷积层，处理用户-特征异质边"""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        
        # 多头注意力参数
        self.W_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_edge = nn.Linear(2, heads, bias=False)  # 边属性投影
        self.W_z = nn.Linear(1, out_channels, bias=False)  # 特征值投影
        
        self.a = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.W_z.weight)
        nn.init.xavier_uniform_(self.a)
        nn.init.zeros_(self.bias)
    
    def forward(self, x_src, x_dst, edge_index, edge_attr, size=None):
        """
        Args:
            x_src: 源节点特征（特征节点）
            x_dst: 目标节点特征（用户节点）
            edge_index: 边索引
            edge_attr: 边属性 [omega, z_val]
        """
        # 多头变换
        h_src = self.W_src(x_src).view(-1, self.heads, self.out_channels)
        h_dst = self.W_dst(x_dst).view(-1, self.heads, self.out_channels)
        
        # 消息传递
        out = self.propagate(edge_index, x=(h_src, h_dst), edge_attr=edge_attr, size=size)
        out = out.mean(dim=1)  # 聚合多头
        out = out + self.bias
        
        return out
    
    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        """计算消息"""
        # 解析边属性
        omega = edge_attr[:, 0:1]  # 门控权重
        z_val = edge_attr[:, 1:2]  # 特征值
        
        # 计算注意力分数
        x_cat = torch.cat([x_i, x_j], dim=-1)
        alpha = (x_cat * self.a).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = alpha.unsqueeze(-1)
        
        # 边权重调制
        edge_weights = self.W_edge(edge_attr).sigmoid()
        alpha = alpha * edge_weights.unsqueeze(-1)
        
        # 构造消息：结合特征嵌入和实际值
        msg = x_i + self.W_z(z_val).unsqueeze(1)
        msg = omega.unsqueeze(1) * msg  # 门控
        msg = alpha * msg
        
        return msg


class CauGramerLayer(nn.Module):
    """CauGramer风格的跨注意力层，用于建模高阶邻居干扰"""
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        # L层GCN用于聚合高阶邻居
        self.gcn_layers = nn.ModuleList([
            GATConv(in_channels if i == 0 else out_channels, 
                   out_channels, heads=1, concat=False)
            for i in range(num_layers)
        ])
        
        # 跨注意力机制
        self.W_q = nn.Linear(out_channels, out_channels)
        self.W_k = nn.Linear(out_channels, out_channels)
        self.W_v = nn.Linear(out_channels, out_channels)
        
        # 主效应和邻居效应分离
        self.main_effect = nn.Linear(out_channels, out_channels)
        self.neighbor_effect = nn.Linear(out_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征
            edge_index: 边索引
        Returns:
            h: 更新后的节点嵌入
            g: 邻居效应向量
        """
        # 多层GCN聚合
        h = x
        neighbor_embeds = []
        
        for i, gcn in enumerate(self.gcn_layers):
            h = gcn(h, edge_index)
            if i > 0:  # 收集高阶邻居表示
                neighbor_embeds.append(h)
            h = F.relu(h)
        
        if len(neighbor_embeds) > 0:
            # 跨注意力：中心节点作为query，邻居聚合作为key/value
            neighbor_h = torch.stack(neighbor_embeds, dim=1).mean(dim=1)
            
            Q = self.W_q(x)
            K = self.W_k(neighbor_h)
            V = self.W_v(neighbor_h)
            
            # 计算注意力
            attn_scores = (Q * K).sum(dim=-1) / math.sqrt(Q.size(-1))
            attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)
            
            # 加权聚合邻居效应
            g = (attn_weights * V).sum(dim=0)
            g = self.neighbor_effect(g)
        else:
            g = torch.zeros_like(x)
        
        # 主效应
        h_main = self.main_effect(x)
        
        # 组合主效应和邻居效应
        h_final = h_main + g
        
        return h_final, g


class HeteroGNNModel(nn.Module):
    """完整的异质图神经网络模型"""
    
    def __init__(self, 
                 user_in_dim: int,
                 feat_embed_dim: int = 32,
                 hidden_dim: int = 128,
                 out_dim: int = 64,
                 num_heads: int = 4,
                 num_cau_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dropout = dropout
        
        # 用户节点初始变换
        self.user_proj = nn.Linear(user_in_dim, hidden_dim)
        
        # 特征节点嵌入已在图构建时初始化
        self.feat_proj = nn.Linear(feat_embed_dim, hidden_dim)
        
        # 异质图卷积层
        self.conv1 = HeteroConv({
            ('feature', 'rev_has', 'user'): RelationalGATConv(
                hidden_dim, hidden_dim, heads=num_heads
            ),
            ('user', 'social', 'user'): GATConv(
                hidden_dim, hidden_dim, heads=num_heads, concat=False
            )
        })
        
        # CauGramer层用于用户-用户子图
        self.cau_layer = CauGramerLayer(hidden_dim, hidden_dim, num_cau_layers)
        
        # DSL嵌入层
        self.dsl_conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.dsl_proj = nn.Linear(16, hidden_dim)
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
    
    def compute_dsl_embedding(self, edge_attr, user_idx, n_users, n_features):
        """计算DSL嵌入"""
        # 构建DSL张量
        dsl_tensor = torch.zeros(n_users, n_features)
        
        # 从边属性中提取omega (1 - omega表示缺失)
        edge_index = edge_attr.shape[0]
        for i in range(edge_index):
            u = user_idx[0, i]
            f = user_idx[1, i]
            omega = edge_attr[i, 0]
            dsl_tensor[u, f] = 1 - omega
        
        # 1D卷积处理
        dsl_tensor = dsl_tensor.unsqueeze(1)  # [N, 1, F]
        dsl_embed = self.dsl_conv(dsl_tensor)  # [N, 16, F]
        dsl_embed = dsl_embed.mean(dim=2)  # [N, 16]
        dsl_embed = self.dsl_proj(dsl_embed)  # [N, hidden_dim]
        
        return dsl_embed
    
    def forward(self, graph: HeteroData) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Returns:
            包含用户嵌入、邻居效应和DSL嵌入的字典
        """
        # 1. 节点特征投影
        x_dict = {
            'user': self.user_proj(graph['user'].x),
            'feature': self.feat_proj(graph['feature'].embed)
        }
        
        # 应用dropout
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) 
                  for k, v in x_dict.items()}
        
        # 2. 第一轮异质图卷积
        edge_index_dict = {
            ('feature', 'rev_has', 'user'): (
                graph['user', 'has', 'feature'].edge_index.flip(0),
                graph['user', 'has', 'feature'].edge_attr
            ),
            ('user', 'social', 'user'): graph['user', 'social', 'user'].edge_index
        }
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # 3. CauGramer层处理用户-用户子图
        if ('user', 'social', 'user') in graph.edge_types:
            user_h, neighbor_g = self.cau_layer(
                x_dict['user'], 
                graph['user', 'social', 'user'].edge_index
            )
        else:
            user_h = x_dict['user']
            neighbor_g = torch.zeros_like(user_h)
        
        # 4. 计算DSL嵌入
        dsl_embed = self.compute_dsl_embedding(
            graph['user', 'has', 'feature'].edge_attr,
            graph['user', 'has', 'feature'].edge_index,
            graph['user'].num_nodes,
            graph['feature'].num_nodes
        )
        
        # 5. 拼接最终表示
        user_final = torch.cat([user_h, neighbor_g, dsl_embed], dim=1)
        user_final = self.output_proj(user_final)
        
        return {
            'user_embed': user_final,
            'neighbor_effect': neighbor_g,
            'dsl_embed': dsl_embed,
            'feature_embed': x_dict.get('feature', None)
        }


class DualTowerPredictor(nn.Module):
    """双塔预测器：Treatment Tower和Outcome Tower"""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Treatment Tower (ZILN)
        self.zero_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.density_estimator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # mu, log_sigma
        )
        
        # Outcome Tower
        self.outcome_predictor = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),  # +1 for treatment L
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_embed, treatment_L):
        """
        Args:
            user_embed: 用户嵌入
            treatment_L: 处理变量（额度）
        Returns:
            p_zero: 零概率
            mu, sigma: 对数正态分布参数
            outcome: 预测结果
        """
        # Treatment Tower
        p_zero = self.zero_classifier(user_embed)
        density_params = self.density_estimator(user_embed)
        mu = density_params[:, 0]
        log_sigma = density_params[:, 1]
        sigma = F.softplus(log_sigma) + 1e-6
        
        # Outcome Tower
        tower_input = torch.cat([user_embed, treatment_L.unsqueeze(1)], dim=1)
        outcome = self.outcome_predictor(tower_input)
        
        return {
            'p_zero': p_zero,
            'mu': mu,
            'sigma': sigma,
            'outcome': outcome
        }


# 使用示例
if __name__ == "__main__":
    # 创建模拟的异质图
    from hetero_graph_builder import HeteroGraphBuilder
    
    # 生成数据
    n_users = 100
    n_features = 10
    user_features = np.random.randn(n_users, n_features)
    user_edges = np.random.randint(0, n_users, size=(2, 200))
    
    # 构建图
    builder = HeteroGraphBuilder()
    graph = builder.build_graph(user_features, user_edges)
    
    # 创建模型
    model = HeteroGNNModel(
        user_in_dim=n_features,
        feat_embed_dim=32,
        hidden_dim=128,
        out_dim=64
    )
    
    # 前向传播
    with torch.no_grad():
        outputs = model(graph)
        print(f"用户嵌入维度: {outputs['user_embed'].shape}")
        print(f"邻居效应维度: {outputs['neighbor_effect'].shape}")
        print(f"DSL嵌入维度: {outputs['dsl_embed'].shape}")
    
    # 双塔预测
    predictor = DualTowerPredictor(embed_dim=64)
    treatment_L = torch.rand(n_users) * 10000  # 模拟额度
    
    predictions = predictor(outputs['user_embed'], treatment_L)
    print(f"\n预测结果:")
    print(f"零概率: {predictions['p_zero'].mean().item():.3f}")
    print(f"结果预测: {predictions['outcome'].mean().item():.3f}")