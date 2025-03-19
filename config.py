import os
from pathlib import Path

# 系统配置
CONFIG = {
    # 数据路径
    'data_dir': './data/photos',
    'output_dir': './output',
    'models_dir': './models',
    
    # 预处理参数
    'image_size': 224,
    'batch_size': 64,
    'num_workers': 4,
    
    # 特征提取和对比学习参数
    'feature_dim': 512,
    'projection_dim': 128,
    'temperature': 0.1,
    'learning_rate': 0.0003,
    'weight_decay': 1e-5,
    'epochs': 20,
    'pretrained_model': 'resnet50',
    
    # 聚类参数
    'n_clusters': 20,  # 初始估计的类别数，会自动优化
    'eps': 0.5,        # DBSCAN的邻域大小参数
    'min_samples': 5,  # DBSCAN的最小样本数
    
    # 对焦检测参数 
    'laplacian_threshold': 100,  # 拉普拉斯算子方差阈值
    'fft_threshold': 10,         # 频域能量比阈值
    
    # 增量学习
    'update_frequency': 100,     # 每添加100张新照片更新一次模型
    
    # 设备配置
    'device': 'cuda',
    'mixed_precision': True,     # 使用混合精度训练
}

# 创建必要的目录
for dir_path in [CONFIG['output_dir'], CONFIG['models_dir']]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)