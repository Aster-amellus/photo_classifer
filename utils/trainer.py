import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

from model.simclr import SimCLR
from utils.preprocessing.dataset import create_data_loaders
from utils.clustering.cluster import PhotoClustering
from utils.focus.focus_detection import FocusDetector

class SimCLRTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.model = SimCLR(config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs']
        )
        self.writer = None
        self.global_step = 0
        
        # 设置混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['mixed_precision'])
        
    def train(self, train_loader, epochs=None):
        """训练SimCLR模型"""
        if epochs is None:
            epochs = self.config['epochs']
            
        # 创建TensorBoard
        log_dir = Path(self.config['output_dir']) / 'logs' / f'run_{time.strftime("%Y%m%d-%H%M%S")}'
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        print(f"开始训练SimCLR模型，共 {epochs} 个epoch")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练一个epoch
            epoch_loss = self._train_epoch(train_loader, epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_model('best_model.pth')
                
            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(f'checkpoint_epoch{epoch+1}.pth')
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Best Loss: {best_loss:.4f}")
            
        # 保存最终模型
        self.save_model('final_model.pth')
        
        if self.writer is not None:
            self.writer.close()
            
        return self.model
    
    def _train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for i, (x_i, x_j, _) in enumerate(pbar):
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)
                
                # 混合精度训练
                with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
                    # 前向传播获取编码后的特征
                    z_i, z_j = self.model(x_i, x_j)
                    
                    # 计算对比损失
                    loss = self.model.contrastive_loss(z_i, z_j)
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 更新进度条
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # 记录到TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                    self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                    
                self.global_step += 1
                
        return epoch_loss / len(train_loader)
    
    def extract_features(self, data_loader):
        """从数据中提取特征"""
        self.model.eval()
        features = []
        image_paths = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                # 如果是元组，第一个元素是图像，第二个元素是路径
                if isinstance(batch, tuple) or isinstance(batch, list):
                    images, paths = batch
                else:
                    images = batch['image']
                    paths = batch['path']
                    
                images = images.to(self.device)
                
                # 提取特征
                batch_features = self.model.forward_features(images)
                
                # 将特征和路径添加到列表中
                features.append(batch_features.cpu().numpy())
                image_paths.extend(paths)
                
        # 合并所有特征
        features = np.vstack(features)
        
        return features, image_paths
    
    def save_model(self, filename):
        """保存模型"""
        models_dir = Path(self.config['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = models_dir / filename
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'config': self.config
        }, save_path)
        
        print(f"Model saved to {save_path}")
        
    def load_model(self, filename):
        """加载模型"""
        load_path = Path(self.config['models_dir']) / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"No model found at {load_path}")
            
        # 加载模型状态
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Model loaded from {load_path}")
        
        return self.model
    
    def visualize_features(self, features, labels, image_paths=None, focus_scores=None, n_samples=1000, method='tsne'):
        """可视化特征分布"""
        # 检查维度一致性
        n_features = len(features)
        n_labels = len(labels)
        n_paths = len(image_paths) if image_paths is not None else 0
        n_scores = len(focus_scores) if focus_scores is not None else 0
        
        if n_features != n_labels:
            print(f"警告: 特征数量 ({n_features}) 与标签数量 ({n_labels}) 不一致")
            # 使用较小的长度
            n_samples = min(n_features, n_labels, n_samples)
        
        if image_paths is not None and n_paths != n_features:
            print(f"警告: 路径数量 ({n_paths}) 与特征数量 ({n_features}) 不一致")
            # 路径数量不足时禁用路径采样
            if n_paths < n_features:
                print("路径数量不足，不进行路径采样")
                image_paths = None
        
        if focus_scores is not None and n_scores != n_features:
            print(f"警告: 对焦分数数量 ({n_scores}) 与特征数量 ({n_features}) 不一致")
            # 分数数量不足时禁用分数采样
            if n_scores < n_features:
                print("对焦分数数量不足，不进行对焦分数采样")
                focus_scores = None
        
        # 如果特征数量太多，随机采样
        if n_features > n_samples:
            print(f"特征数量 ({n_features}) 超过采样限制 ({n_samples})，进行随机采样")
            # 生成随机索引，确保不超出最小长度
            max_index = min(n_features, n_labels) - 1
            indices = np.random.choice(max_index + 1, min(n_samples, max_index + 1), replace=False)
            
            sampled_features = features[indices]
            sampled_labels = labels[indices]
            
            # 安全地采样路径和分数
            if image_paths is not None:
                # 确保索引不超过路径数组长度
                valid_indices = [i for i in indices if i < len(image_paths)]
                sampled_paths = [image_paths[i] for i in valid_indices] if valid_indices else None
            else:
                sampled_paths = None
                
            if focus_scores is not None:
                # 确保索引不超过分数数组长度
                sampled_focus_scores = focus_scores[indices] if all(i < len(focus_scores) for i in indices) else None
            else:
                sampled_focus_scores = None
        else:
            sampled_features = features
            sampled_labels = labels
            sampled_paths = image_paths
            sampled_focus_scores = focus_scores
        
        # 降维
        if method == 'tsne':
            print("使用t-SNE进行降维...")
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            print("使用PCA进行降维...")
            reducer = PCA(n_components=2, random_state=42)
                
        # 安全地进行降维
        try:
            embedded = reducer.fit_transform(sampled_features)
        except Exception as e:
            print(f"降维失败: {e}")
            print("尝试减少样本数量或使用PCA方法")
            return
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        # 确定点大小
        point_sizes = 50
        if sampled_focus_scores is not None:
            # 归一化分数并调整大小
            min_size, max_size = 20, 100
            norm_scores = np.clip(sampled_focus_scores, 0, 100) / 100
            point_sizes = min_size + norm_scores * (max_size - min_size)
        
        scatter = plt.scatter(
            embedded[:, 0], 
            embedded[:, 1], 
            c=sampled_labels, 
            cmap='tab20', 
            alpha=0.7,
            s=point_sizes
        )
        
        # 添加图例
        unique_labels = np.unique(sampled_labels)
        legend1 = plt.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), 
                               markersize=8, label=f'Cluster {label}') for label in unique_labels],
            title="Clusters",
            loc="upper right"
        )
        plt.gca().add_artist(legend1)
        
        # 如果有对焦分数，添加大小图例
        if sampled_focus_scores is not None:
            sizes = [20, 50, 80]
            labels = ['Low Focus', 'Medium Focus', 'High Focus']
            legend2 = plt.legend(
                handles=[plt.Line2D([0], [0], marker='o', color='gray', 
                                   markersize=size/10, label=label) for size, label in zip(sizes, labels)],
                title="Focus Quality",
                loc="lower right"
            )
            plt.gca().add_artist(legend2)
        
        plt.title(f'Feature Visualization ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        
        # 保存图像
        output_dir = Path(self.config['output_dir']) / 'visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'feature_visualization_{method}.png', dpi=300)
        plt.close()
        
        # 如果有图像路径，保存交互式HTML可视化（如果可能）
        if sampled_paths and len(sampled_paths) > 0:
            try:
                import plotly.express as px
                
                # 提取文件名
                file_names = [Path(p).name for p in sampled_paths]
                
                # 创建数据框
                df_data = {
                    'x': embedded[:, 0],
                    'y': embedded[:, 1],
                    'cluster': sampled_labels,
                    'file': file_names
                }
                
                if sampled_focus_scores is not None:
                    df_data['focus_score'] = sampled_focus_scores
                    
                df = pd.DataFrame(df_data)
                
                # 创建Plotly图
                fig = px.scatter(
                    df, x='x', y='y', 
                    color='cluster', 
                    hover_data=['file'] + (['focus_score'] if sampled_focus_scores is not None else []),
                    size='focus_score' if sampled_focus_scores is not None else None,
                    size_max=15,
                    title=f'Interactive Feature Visualization ({method.upper()})'
                )
                
                # 保存为HTML
                fig.write_html(output_dir / f'interactive_visualization_{method}.html')
                print(f"交互式可视化已保存到: {output_dir / f'interactive_visualization_{method}.html'}")
                
            except ImportError:
                print("plotly库未安装，跳过交互式可视化")
            except Exception as e:
                print(f"创建交互式可视化时出错: {e}")