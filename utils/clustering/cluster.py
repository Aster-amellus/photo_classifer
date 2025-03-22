import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import shutil
import time
from collections import Counter

class PhotoClustering:
    def __init__(self, config):
        self.config = config
        self.kmeans = None
        self.dbscan = None
        self.optimized_clusters = False
        self.n_clusters = config['n_clusters']
        self.eps = config['eps']
        self.min_samples = config['min_samples']
        
    def optimize_clusters(self, features, max_clusters=50, min_clusters=5, step=5):
        """优化聚类数量"""
        print("优化聚类数量...")
        
        k_range = range(min_clusters, max_clusters+1, step)
        sil_scores = []
        db_scores = []
        ch_scores = []
        
        for k in tqdm(k_range, desc="Finding optimal number of clusters"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # 计算评价指标
            sil = silhouette_score(features, labels)
            db = davies_bouldin_score(features, labels)
            ch = calinski_harabasz_score(features, labels)
            
            sil_scores.append(sil)
            db_scores.append(db)
            ch_scores.append(ch)  # 不需要归一化，直接使用原始值
        
        # 找到最佳聚类数
        best_k_sil = k_range[np.argmax(sil_scores)]
        best_k_db = k_range[np.argmin(db_scores)]
        best_k_ch = k_range[np.argmax(ch_scores)]
        
        # 计算加权平均
        weights = [0.4, 0.3, 0.3]  # 权重可调整
        best_k = int(np.round(weights[0] * best_k_sil + weights[1] * best_k_db + weights[2] * best_k_ch))
        
        print(f"优化结果 - 轮廓系数最佳K: {best_k_sil}, 戴维斯-博尔丁最佳K: {best_k_db}, CH指数最佳K: {best_k_ch}")
        print(f"最终选择的聚类数量: {best_k}")
        
        # 绘制评估指标图
        self._plot_cluster_metrics(k_range, sil_scores, db_scores, ch_scores, best_k)
        
        self.n_clusters = best_k
        self.optimized_clusters = True
        return best_k
    
    def _plot_cluster_metrics(self, k_range, sil_scores, db_scores, ch_scores, best_k):
        """绘制聚类评价指标"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 轮廓系数 (越高越好)
        ax1.plot(k_range, sil_scores, 'o-')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score vs. Number of Clusters')
        ax1.axvline(x=best_k, color='r', linestyle='--')
        
        # 戴维斯-博尔丁指数 (越低越好)
        ax2.plot(k_range, db_scores, 'o-')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('Davies-Bouldin Index vs. Number of Clusters')
        ax2.axvline(x=best_k, color='r', linestyle='--')
        
        # Calinski-Harabasz指数 (越高越好)
        ax3.plot(k_range, ch_scores, 'o-')
        ax3.set_xlabel('Number of clusters')
        ax3.set_ylabel('Calinski-Harabasz Index')
        ax3.set_title('Calinski-Harabasz Index vs. Number of Clusters')
        ax3.axvline(x=best_k, color='r', linestyle='--')
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path(self.config['output_dir']) / 'clustering'
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'cluster_metrics.png')
        plt.close()
    
    def fit_kmeans(self, features):
        """训练K-means聚类模型"""
        print(f"使用K-means进行聚类, k={self.n_clusters}...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans_labels = self.kmeans.fit_predict(features)
        return kmeans_labels
    
    def fit_dbscan(self, features):
        """训练DBSCAN聚类模型"""
        print(f"使用DBSCAN进行聚类, eps={self.eps}, min_samples={self.min_samples}...")
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        dbscan_labels = self.dbscan.fit_predict(features)
        return dbscan_labels
    
    def fit(self, features, optimize=True):
        """训练聚类模型"""
        if optimize and not self.optimized_clusters:
            self.optimize_clusters(features)
        
        # 训练KMeans
        kmeans_labels = self.fit_kmeans(features)
        
        # 训练DBSCAN
        dbscan_labels = self.fit_dbscan(features)
        
        # 合并聚类结果
        labels = self._combine_clustering_results(features, kmeans_labels, dbscan_labels)
        
        return labels
    
    def _combine_clustering_results(self, features, kmeans_labels, dbscan_labels):
        """合并KMeans和DBSCAN的聚类结果"""
        # 统计每个K-means簇中的DBSCAN噪声点数量
        noise_mask = dbscan_labels == -1
        
        # 如果DBSCAN没有发现任何噪声点，直接返回K-means结果
        if not np.any(noise_mask):
            return kmeans_labels
            
        # 计算每个K-means簇的质量
        unique_kmeans = np.unique(kmeans_labels)
        cluster_quality = {}
        
        for cluster in unique_kmeans:
            cluster_mask = kmeans_labels == cluster
            cluster_points = features[cluster_mask]
            
            # 如果簇太小，不计算质量
            if len(cluster_points) < 5:
                cluster_quality[cluster] = 0
                continue
                
            # 计算簇内点的平均距离
            centroid = self.kmeans.cluster_centers_[cluster]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            cluster_quality[cluster] = 1.0 / (np.mean(distances) + 1e-10)
        
        # 合并结果：将DBSCAN识别为噪声的点分配给最近的高质量K-means簇
        combined_labels = kmeans_labels.copy()
        
        for idx in np.where(noise_mask)[0]:
            point = features[idx].reshape(1, -1)
            
            # 计算到所有K-means中心的距离
            distances = np.array([
                np.linalg.norm(point - self.kmeans.cluster_centers_[c]) / (cluster_quality.get(c, 1e-10) + 1e-10)
                for c in unique_kmeans
            ])
            
            # 分配给质量加权距离最近的簇
            combined_labels[idx] = unique_kmeans[np.argmin(distances)]
        
        return combined_labels
    
    def predict(self, features):
        """预测新样本的聚类标签"""
        if self.kmeans is None:
            raise ValueError("模型尚未训练")
        
        # 使用K-means预测
        return self.kmeans.predict(features)
    
    def save(self, filepath):
        """保存聚类模型"""
        model_data = {
            'kmeans': self.kmeans,
            'dbscan': self.dbscan,
            'n_clusters': self.n_clusters,
            'eps': self.eps,
            'min_samples': self.min_samples,
            'optimized_clusters': self.optimized_clusters
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load(self, filepath):
        """加载聚类模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.kmeans = model_data['kmeans']
        self.dbscan = model_data['dbscan']
        self.n_clusters = model_data['n_clusters']
        self.eps = model_data['eps']
        self.min_samples = model_data['min_samples']
        self.optimized_clusters = model_data['optimized_clusters']
        
    def organize_photos(self, image_paths, labels, focus_scores=None):
        """根据聚类结果组织照片"""
        # 创建输出目录
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(self.config['output_dir']) / f'organized_photos_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有提供对焦分数，则创建一个默认的最佳对焦分数
        if focus_scores is None:
            focus_scores = np.ones(len(image_paths)) * 100  # 设置为最大对焦分数
        
        # 查找每个类别的照片数量
        cluster_counts = Counter(labels)
        
        # 按照照片数量对类别进行排序（从大到小）
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 创建映射字典，将原始标签映射到新的有序标签
        cluster_mapping = {old_label: i for i, (old_label, _) in enumerate(sorted_clusters)}
        
        # 创建复制任务列表
        copy_tasks = []
        
        for path, label, focus_score in zip(image_paths, labels, focus_scores):
            # 获取新的有序标签
            new_label = cluster_mapping[label]
            
            # 确定目标目录（是否对焦良好）
            if focus_score >= 50:  # 对焦良好的阈值
                target_dir = output_dir / f'cluster_{new_label:03d}'
            else:
                target_dir = output_dir / f'cluster_{new_label:03d}' / 'low_focus'
            
            # 创建目标目录
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 目标文件路径
            target_path = target_dir / Path(path).name
            
            # 添加到复制任务列表
            copy_tasks.append((path, target_path))
        
        # 执行复制任务
        for src, dst in copy_tasks:
            shutil.copy2(src, dst)
        
        return output_dir