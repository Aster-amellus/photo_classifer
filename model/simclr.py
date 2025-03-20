import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import deque
import random

class SimCLR(nn.Module):
    def __init__(self, config):
        super(SimCLR, self).__init__()
        
        # 获取配置
        self.use_moco = config.get('use_moco', False)
        self.momentum = config.get('momentum', 0.999)
        self.queue_size = config.get('queue_size', 4096)
        
        # 选择backbone
        self.encoder = self._get_encoder(config['pretrained_model'])
        
        # 获取backbone的输出维度
        self.feature_dim = self._get_feature_dim(config['pretrained_model'])
        
        # 投影头 - 更复杂的多层MLP
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 2, config['projection_dim'])
        )
        
        # 如果使用MoCo，创建动量编码器和队列
        if self.use_moco:
            # 动量编码器
            self.encoder_k = self._get_encoder(config['pretrained_model'])
            self.projector_k = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.BatchNorm1d(self.feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim // 2, config['projection_dim'])
            )
            
            # 初始化动量编码器权重，不需要梯度
            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
                
            for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            
            # 创建队列
            self.register_buffer("queue", torch.randn(config['projection_dim'], self.queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.temperature = config['temperature']
        
    def _get_encoder(self, model_name):
        """获取预训练的特征提取器，支持更多模型"""
        if model_name == 'resnet50':
            # 使用新的API加载预训练模型
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(model.children())[:-1])
        elif 'efficientnet' in model_name:
            if model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif model_name == 'efficientnet_b1':
                model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            elif model_name == 'efficientnet_b2':
                model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            elif model_name == 'efficientnet_b3':
                model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            # 移除分类头
            features = model.features
            pool = model.avgpool
            return nn.Sequential(features, pool)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _get_feature_dim(self, model_name):
        """获取特征提取器输出的维度"""
        if 'resnet50' in model_name or 'resnet101' in model_name:
            return 2048
        elif 'resnet18' in model_name:
            return 512
        elif 'efficientnet_b0' in model_name:
            return 1280
        elif 'efficientnet_b1' in model_name:
            return 1280
        elif 'efficientnet_b2' in model_name:
            return 1408
        elif 'efficientnet_b3' in model_name:
            return 1536
        else:
            return 2048  # 默认值
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """MoCo中更新动量编码器"""
        if not self.use_moco:
            return
            
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            
        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """MoCo中更新队列"""
        if not self.use_moco:
            return
            
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的keys
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 处理循环队列末尾
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size-remaining] = keys[remaining:].T
            
        # 移动指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward_features(self, x):
        """仅通过encoder提取特征"""
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        return features
        
    def forward_k(self, x):
        """使用动量编码器前向传播(仅MoCo使用)"""
        if not self.use_moco:
            return None
            
        with torch.no_grad():
            features_k = self.encoder_k(x)
            features_k = torch.flatten(features_k, start_dim=1)
            proj_k = self.projector_k(features_k)
            proj_k = F.normalize(proj_k, dim=1)
            return proj_k
        
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
        
        # 投影到对比空间
        p1 = self.projector(z1)
        p1 = F.normalize(p1, dim=1)
        
        if self.use_moco:
            # 动量编码器处理第二个视图
            self._momentum_update_key_encoder()
            p2 = self.forward_k(x2)
            
            # 入队列
            self._dequeue_and_enqueue(p2)
            
            return p1, p2
        else:
            # 原始SimCLR方式
            z2 = self.forward_features(x2)
            p2 = self.projector(z2)
            p2 = F.normalize(p2, dim=1)
            return p1, p2
        
    def contrastive_loss(self, p1, p2):
        """计算对比损失，支持MoCo和SimCLR"""
        if self.use_moco:
            # MoCo风格的InfoNCE损失
            # 正样本对
            l_pos = torch.einsum('nc,nc->n', [p1, p2]).unsqueeze(-1)
            
            # 负样本：使用队列
            l_neg = torch.einsum('nc,ck->nk', [p1, self.queue.clone().detach()])
            
            # 逻辑回归
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            
            # 标签：正样本总是第一个
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=p1.device)
            
            loss = F.cross_entropy(logits, labels)
            return loss
        else:
            # 原始NT-Xent损失
            batch_size = p1.shape[0]
            
            # 计算所有样本对之间的相似度矩阵
            z = torch.cat([p1, p2], dim=0)
            sim = torch.mm(z, z.t().contiguous()) / self.temperature
            
            # 掩码排除自身匹配
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