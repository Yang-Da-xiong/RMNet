import torch
import torch.nn as nn
import torch.nn.functional as F

#聚类损失
class Feature_Clustering_Loss(nn.Module):
    def __init__(self, num_class=4, feat_dim=128):
        super(Feature_Clustering_Loss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to('cuda'))

    def forward(self, x, labels):
        #x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to('cuda')
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


#距离约束
class Distance_Constraint_Loss(nn.Module):
    def __init__(self, temperature=0.5):
        super(Distance_Constraint_Loss, self).__init__()
        self.temperature = temperature
        self.gap = nn.AdaptiveAvgPool2d(1)  #自适应平均池化

    def forward(self, features, labels):
        #features = self.gap(features).view(features.size(0), -1)  #[N,C,H,W]  #[N,C]
        #features = features.view(features.shape[0],-1)
        #print(features.shape)
        #print(labels.shape)
        batch_size = features.shape[0]
        features = features / features.norm(dim=1, keepdim=True)  #规范化处理，这样会使L2范数为1
        # 计算余弦相似度
        #print(features.unsqueeze(1).shape) #[32,1,128] [32,32,128]
        #print(features.unsqueeze(0).shape) #[1,32,128] [32,32,128]
        #存在广播机制，使得得到的shape是sim_matrix

        #余弦相似度
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / self.temperature

        #矩阵点乘
        #sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        '''
        #欧氏距离
        squared_features = torch.sum(features ** 2, dim=1, keepdim=True)
        # 2. 计算特征矩阵与其转置的内积 (64, 64)
        sim_matrix = torch.matmul(features, features.T)
        # 3. 利用广播机制计算欧氏距离
        distance_matrix = squared_features + squared_features.T - 2 * sim_matrix
        # 4. 为了避免数值问题，确保距离矩阵没有负数（因为浮动误差可能导致微小的负数）
        distance_matrix = torch.maximum(distance_matrix, torch.zeros_like(distance_matrix))
        # 5. 对欧氏距离进行开方
        sim_matrix = torch.sqrt(distance_matrix)
        '''
        #  0    1   2   3
        #0 0.7  0.6
        #1
        #2
        #3
        # Mask to remove the diagonal elements from the positive term
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)  #创建对角矩阵
        
        # 计算同类样本之间的相似度的指数形式，这里不太对，同类样本应该是距离相近的
        #print(labels.unsqueeze(1) == labels.unsqueeze(0))  计算的样本i与样本j之间的标签是否相同
        exp_sim_same = torch.exp(sim_matrix) * (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        #[~mask]去除对角线元素
        exp_sim_same = exp_sim_same[~mask].view(batch_size, -1).sum(dim=1)  #如果i==j，代表是同一个样本，不进行计算

        # 计算所有样本之间的相似度的指数形式
        exp_sim_all = torch.exp(sim_matrix)
        exp_sim_all = exp_sim_all[~mask].view(batch_size, -1)
        exp_sim_all = exp_sim_all.sum(dim=1)
        
        # 计算损失
        loss = -torch.log(exp_sim_same / exp_sim_all).mean()
        
        return loss
    