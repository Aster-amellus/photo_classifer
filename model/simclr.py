import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, config):
        super(SimCLR, self).__init__()
        
        # 选择backbone
        self.encoder = self._get_encoder(config['pretrained_model'])
        
        # 获取backbone的输出维度
        self.feature_dim = self._get_feature_dim()
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, config['projection_dim'])
        )
        
        self.temperature = config['temperature']
        
    def _get_encoder(self, model_name):
        """获取预训练的特征提取器"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # 移除最后的全连接层
            return nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            return nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _get_feature_dim(self):
        """获取特征提取器输出的维度"""
        if hasattr(self.encoder[-1], 'num_features'):
            return self.encoder[-1].num_features
        # 对于ResNet，返回最后一层的输出维度
        return 2048 if isinstance(self.encoder[-1], nn.AdaptiveAvgPool2d) else 512
    
    def forward_features(self, x):
        """仅通过encoder提取特征"""
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        return features
        
    def forward(self, x1, x2=None):
        """
        如果提供两个视图，返回投影后的特征用于对比学习
        如果只提供一个输入，只返回编码后的特征
        """
        if x2 is None:
            # 特征提取模式
            return self.forward_features(x1)
            
        # 对比学习模式
        z1 = self.forward_features(x1)
        z2 = self.forward_features(x2)
        
        # 投影到对比空间
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        
        # L2归一化
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        
        return p1, p2
        
    def contrastive_loss(self, p1, p2):
        """计算NT-Xent对比损失"""
        batch_size = p1.shape[0]
        
        # 计算所有样本对之间的相似度矩阵
        z = torch.cat([p1, p2], dim=0)
        sim = torch.mm(z, z.t().contiguous()) / self.temperature
        
        # 把对角线元素设置为很小的值，避免自身匹配
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        # 正样本对的相似度
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
        
        # 创建掩码移除自身匹配
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool, device=sim.device)
        mask = mask.fill_diagonal_(0)
        
        # 所有负样本对的相似度
        negative_samples = sim[mask].reshape(2 * batch_size, -1)
        
        # 用softmax计算概率
        labels = torch.zeros(2 * batch_size, device=sim.device, dtype=torch.long)
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss