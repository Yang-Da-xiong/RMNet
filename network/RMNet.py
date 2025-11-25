import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from layers import GraphAttentionLayer
from torch_geometric.nn import APPNP, global_mean_pool

class SpatialCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        # 为Q、K、V创建独立的线性变换
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影层
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v):
        # q: (B, C, H, W)  - GAT特征
        # k, v: (B, C, H, W) - CNN特征
        
        # 应用投影变换
        query = self.query_proj(q)  # (B, C, H, W)
        key = self.key_proj(k)      # (B, C, H, W)
        value = self.value_proj(v)  # (B, C, H, W)
        
        # 重新排列维度：(B, C, H, W) -> (B, num_heads, head_dim, H*W)
        batch_size, _, height, width = query.shape
        query = query.view(batch_size, self.num_heads, self.head_dim, -1)
        key = key.view(batch_size, self.num_heads, self.head_dim, -1)
        value = value.view(batch_size, self.num_heads, self.head_dim, -1)
        
        # 计算注意力得分：QK^T / sqrt(d_k)
        attn_scores = torch.matmul(
            query.transpose(2, 3),  # (B, num_heads, H*W, head_dim)
            key                    # (B, num_heads, head_dim, H*W)
        ) * self.scale  # (B, num_heads, H*W, H*W)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 注意力加权：权重 * V
        attn_output = torch.matmul(
            attn_weights,  # (B, num_heads, H*W, H*W)
            value.transpose(2, 3)  # (B, num_heads, H*W, head_dim)
        )  # (B, num_heads, H*W, head_dim)
        
        # 调整维度并合并多头
        attn_output = attn_output.transpose(2, 3).contiguous()  # (B, num_heads, head_dim, H*W)
        attn_output = attn_output.view(batch_size, -1, height, width)  # (B, C, H, W)
        
        # 输出投影
        attn_output = self.output_proj(attn_output)  # (B, C, H, W)
        
        return attn_output

#MLP层
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256, dropout_rate=0.0):
        """
        简化的MLP层：升维 → 激活 → (可选Dropout) → 降维
        
        参数:
            input_dim: 输入特征维度 (256)
            hidden_dim: 隐藏层维度 (升维后的维度)
            output_dim: 输出特征维度
            dropout_rate: Dropout比率
        """
        super(SimpleMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 升维
        self.activation = nn.ReLU()                   # 激活函数
        #self.dropout = nn.Dropout(dropout_rate)       # Dropout (可选)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 降维
        
    def forward(self, x):
        x = self.fc1(x)          # 升维: (32, 256) → (32, 512)
        x = self.activation(x)    # 激活
        #x = self.dropout(x)       # Dropout (训练时有效，评估时自动跳过)
        x = self.fc2(x)           # 降维: (32, 512) → (32, 128)
        return x

# 支持边权重的APPNP模型
class WeightedGNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(WeightedGNNModel, self).__init__()
        self.conv1 = APPNP(K=1, alpha=0.3)
        self.conv2 = APPNP(K=1, alpha=0.3)
        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        支持带权重的前向传播
        Args:
            x: 节点特征 [N, F]
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E,] (可选)
            batch: 批索引 [N,] (可选)
        """
        # 带权重的APPNP传播
        x = F.sigmoid(self.conv1(x, edge_index, edge_weight))
        x = F.sigmoid(self.conv2(x, edge_index, edge_weight))
        
        # 池化层（如果提供批索引）
        if batch is not None:
            x = self.pool(x, batch)
        return x

class HybridAdjacency:
    """融合半径邻接与特征相似度的混合邻接矩阵生成器"""
    
    def __init__(self, grid_size=16, radius=3, feat_channels=256, device='cuda'):
        """
        Args:
            grid_size: 特征图尺寸 (N*N)
            radius: 强连接半径
            feat_channels: 特征通道数
            device: 计算设备
        """
        self.grid_size = grid_size
        self.radius = radius
        self.feat_channels = feat_channels
        self.device = device
        
        # 预计算半径邻接矩阵（二值）
        self.radius_adj = self._create_radius_adjacency()
    
    def _create_radius_adjacency(self):
        """创建半径邻接矩阵 (N*N)"""
        total_nodes = self.grid_size * self.grid_size
        adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        
        kernel_size = 2 * self.radius + 1
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current_idx = i * self.grid_size + j
                
                # 计算当前像素的邻域边界
                i_min = max(0, i - self.radius)
                i_max = min(self.grid_size, i + self.radius + 1)
                j_min = max(0, j - self.radius)
                j_max = min(self.grid_size, j + self.radius + 1)
                
                # 标记邻域内所有节点
                for ni in range(i_min, i_max):
                    for nj in range(j_min, j_max):
                        neighbor_idx = ni * self.grid_size + nj
                        adj[current_idx, neighbor_idx] = 1
        
        # 添加自环
        np.fill_diagonal(adj, 1)
        return torch.tensor(adj, device=self.device, dtype=torch.float32)
    
    def build_hybrid_adj(self, feature_map, sim_threshold=0.8):
        """
        构建融合邻接矩阵
        Args:
            feature_map: 节点特征 (B, N, C)
            sim_threshold: 相似度阈值
        Returns:
            batch_edges: 各样本的边索引和边权重列表
        """
        B, N, C = feature_map.shape
        
        # 1. 计算全图余弦相似度
        norm_feat = F.normalize(feature_map, p=2, dim=-1)
        cosine_adj = torch.bmm(norm_feat, norm_feat.transpose(1, 2))  # (B, N, N)
        cosine_adj = 0.5 * (cosine_adj + 1)  # 映射到[0,1]
        
        # 2. 创建局部强连接掩码
        # 扩展半径邻接矩阵到batch维度
        radius_mask = self.radius_adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        
        # 3. 融合两种邻接矩阵
        # 3a. 在半径范围内强制设置强连接
        hybrid_adj = torch.where(
            radius_mask.bool(), 
            torch.ones_like(cosine_adj),  # 强制为1
            cosine_adj                    # 保留相似度
        )
        
        # 3b. 应用全局相似度阈值过滤
        hybrid_adj = torch.where(
            hybrid_adj >= sim_threshold,
            hybrid_adj,
            torch.zeros_like(hybrid_adj)
        )
        
        # 4. 确保自环存在
        diag_mask = torch.eye(N, device=self.device, dtype=torch.bool).unsqueeze(0)
        hybrid_adj = torch.where(diag_mask, torch.ones_like(hybrid_adj), hybrid_adj)
        
        #print(hybrid_adj)
        #with open('tensor_dump.txt', 'w') as f:
        #    f.write(str(hybrid_adj))
        
        # 5. 转换为稀疏COO格式
        batch_edges = []
        for b in range(B):
            sparse_adj = hybrid_adj[b]
            
            # 转换为COO格式
            coo = sparse_adj.to_sparse_coo()
            indices = coo.indices()
            values = coo.values()
            
            batch_edges.append((indices, values))
        
        return batch_edges

'''LWENet'''
class L2_nrom(nn.Module):
    def __init__(self,mode='l2'):
        super(L2_nrom, self).__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True)).pow(0.5)
            norm = embedding / (embedding.pow(2).mean(dim=1, keepdim=True)).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x)
            embedding = _x.sum((2,3), keepdim=True)
            norm = embedding / (torch.abs(embedding).mean(dim=1, keepdim=True))
        return norm

class Sepconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sepconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)

    def forward(self, input):
        out1 = self.conv1(input)
        out = self.conv2(out1)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features,
                                           num_input_features,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, num_input_features,
                                           kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, prev_features):
        new_features = self.conv1(self.relu1(self.norm1(prev_features)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock_Add(nn.Module):
    def __init__(self, num_layers, num_input_features):
        super(_DenseBlock_Add, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = init_features
        for name, layer in self.named_children():
            new_features = layer(features)
            features = features + new_features
        return features


class DenseNet_Add_1(nn.Module):
    def __init__(self, num_layers=6):
        super(DenseNet_Add_1, self).__init__()

        # 高通滤波 卷积核权重初始化
        SRM_npy = np.load('SRM_Kernels.npy')
        self.srm_filters_weight = nn.Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=False)
        self.srm_filters_weight.data.numpy()[:] = SRM_npy

        self.features = nn.Sequential(OrderedDict([('norm0', nn.BatchNorm2d(30)), ]))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        self.block = _DenseBlock_Add(
            num_layers=num_layers,
            num_input_features=30,)

        num_features = 30

        self.trans = _Transition(num_input_features=num_features,
                            num_output_features=32)   # BlockB

    def forward(self, input):
        HPF_output = F.conv2d(input, self.srm_filters_weight, stride=1, padding=2)
        output_shallow = self.features(HPF_output)
        output = self.block(output_shallow)
        output = self.trans(output)
        return output, output_shallow


class RMNet(nn.Module):
    def __init__(self):
        super(RMNet, self).__init__()
        #preprocessing+BlockB
        self.Dense_layers  = DenseNet_Add_1(num_layers=6)

        #feature extraction
        self.layer5 = nn.Conv2d(32, 32, kernel_size=3, padding=1) #BlockC
        self.layer5_BN = nn.BatchNorm2d(32)
        self.layer5_AC = nn.ReLU()

        self.layer6 = nn.Conv2d(32, 64, kernel_size=3, padding=1)#BlockC
        self.layer6_BN = nn.BatchNorm2d(64)
        self.layer6_AC = nn.ReLU()

        self.avgpooling2 = nn.AvgPool2d(kernel_size=3, stride=2,padding=1)

        self.layer7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#BlockC
        self.layer7_BN = nn.BatchNorm2d(64)
        self.layer7_AC = nn.ReLU()

        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, padding=1)#BlockC
        self.layer8_BN  = nn.BatchNorm2d(128)
        self.layer8_AC = nn.ReLU()

        self.avgpooling3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)#BlockC
        self.layer9_BN = nn.BatchNorm2d(128)
        self.layer9_AC = nn.ReLU()

        self.layer10 = Sepconv(128,256) #BlockD
        self.layer10_BN = nn.BatchNorm2d(256)
        self.layer10_AC =  nn.ReLU()

        #MGP
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.L2_norm = L2_nrom(mode='l2')
        self.L1_norm = L2_nrom(mode='l1')
        #classifier
        self.fc1 = nn.Linear(256*3 + 256, 2)

        #APPNP
        self.gcn = WeightedGNNModel(in_channels=256, hidden_channels=512)

        self.create_weighted_adj_R = HybridAdjacency(grid_size = 16, radius = 5, feat_channels=256)

        #MLP
        self.MLP = SimpleMLP(input_dim=256, hidden_dim=512, output_dim=256)

        #cross_attention
        # 在初始化时定义通道调整卷积层
        self.shallow_channel_adjust = nn.Conv2d(32, 256, kernel_size=1)
        self.shallow_adaptive_pool = nn.AdaptiveAvgPool2d((32,32))
        self.cross_att = SpatialCrossAttention(in_channels=256)

    def forward(self, input):

        Dense_block_out, output_shallow = self.Dense_layers(input)
        output_shallow = Dense_block_out  #(32,32,128,128)
        layer5_out = self.layer5(Dense_block_out)
        layer5_out = self.layer5_BN(layer5_out)
        layer5_out = self.layer5_AC(layer5_out)

        layer6_out = self.layer6(layer5_out)
        layer6_out = self.layer6_BN(layer6_out)
        layer6_out = self.layer6_AC(layer6_out)

        avg_pooling2 = self.avgpooling2(layer6_out)

        layer7_out = self.layer7(avg_pooling2)
        layer7_out = self.layer7_BN(layer7_out)
        layer7_out = self.layer7_AC(layer7_out)

        layer8_out = self.layer8(layer7_out)
        layer8_out = self.layer8_BN(layer8_out)
        layer8_out = self.layer8_AC(layer8_out)

        avg_pooling3 = self.avgpooling2(layer8_out)

        layer9_out = self.layer9(avg_pooling3)
        layer9_out = self.layer9_BN(layer9_out)
        layer9_out = self.layer9_AC(layer9_out)

        layer10_out = self.layer10(layer9_out)
        layer10_out = self.layer10_BN(layer10_out)
        layer10_out = self.layer10_AC(layer10_out)
        output_deep = layer10_out

        #avg_pooling4 = self.avgpooling2(layer10_out)  #[32, 256, 16, 16]

        # 浅层调整通道数
        shallow_feat_adjusted = self.shallow_adaptive_pool(output_shallow)
        shallow_feat_adjusted = self.shallow_channel_adjust(shallow_feat_adjusted)
        
        # 应用跨尺度注意力
        attn_output = self.cross_att(q=layer10_out, k=shallow_feat_adjusted, v=shallow_feat_adjusted)
        attn_output = self.avgpooling2(attn_output)

        #feats_gat.shape(32,256,16,16)
        feats_gat = attn_output.view(attn_output.size(0), 256, -1)
        feats_gat = feats_gat.permute(0,2,1)

        batch_edges = self.create_weighted_adj_R.build_hybrid_adj(feats_gat)
        #print(batch_edges.shape)
        num_nodes = feats_gat.size(1)  # 每张图的节点数
        
        # 对每张图进行APPNP传播
        outputs = []
        for i in range(feats_gat.size(0)):
            edge_index, edge_weight = batch_edges[i]
            #print(edge_index.shape)
            
            # 带权重的GNN前向
            graph_out = self.gcn(
                x=feats_gat[i], 
                edge_index=edge_index, 
                edge_weight=edge_weight,
                batch=None
            )
            outputs.append(graph_out)

        # 将输出合并成一个批次数据
        GAT_outputs = torch.mean(torch.stack(outputs, dim=0),dim=1)  # (32, 256)
        GAT_outputs_MLP = self.MLP(GAT_outputs)

        #deep
        output_GAP_deep = self.GAP(layer10_out)
        output_L2_deep = self.L2_norm(layer10_out)
        output_L1_deep = self.L1_norm(layer10_out)

        output_GAP_deep = output_GAP_deep.view(-1, 256)
        output_L2_deep = output_L2_deep.view(-1, 256)
        output_L1_deep = output_L1_deep.view(-1, 256)
        final_deep = torch.cat([output_GAP_deep, output_L2_deep, output_L1_deep], dim = -1)

        Final_feat = torch.cat([final_deep, GAT_outputs_MLP], dim=-1)

        output = self.fc1(Final_feat)
        #print(layer10_out.shape)  #32, 256, 32, 32
        #print("output_L2.shape: ", output_L2.shape)  #[32, 256, 1, 1]

        return output, GAT_outputs_MLP