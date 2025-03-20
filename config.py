import os
from pathlib import Path

# 系统配置
CONFIG = {
    # 数据路径
    'data_dir': './data/photos',
    'output_dir': './output',
    'models_dir': './models',
    
    # 预处理参数
    'image_size': 384,  # 增大图像尺寸，保留更多细节
    'batch_size': 16,   # 根据GPU内存调整
    'num_workers': 4,
    
    # 特征提取和对比学习参数
    'feature_dim': 2048,         # 增加特征维度
    'projection_dim': 256,       # 增加投影维度
    'temperature': 0.07,         # 降低温度参数，增强对比度
    'learning_rate': 0.0003,     # 调整学习率
    'weight_decay': 1e-4,        # 增加权重衰减，减少过拟合
    'epochs': 15,                # 增加训练轮数
    'pretrained_model': 'efficientnet_b3',  # 使用更强大的模型
    'use_moco': True,            # 使用MoCo对比学习框架
    'queue_size': 4096,          # MoCo队列大小
    'momentum': 0.999,           # MoCo动量编码器动量
    
    # 聚类参数
    'n_clusters': 100,  # 初始估计的类别数，会自动优化
    'eps': 0.5,        # DBSCAN的邻域大小参数
    'min_samples': 5,  # DBSCAN的最小样本数
    'use_hierarchical': True,  # 使用层次聚类优化结果
    
    # 对焦检测参数 
    'laplacian_threshold': 200,   # 调整拉普拉斯算子方差阈值
    'fft_threshold': 20,          # 调整频域能量比阈值
    'use_deep_focus': True,       # 使用深度学习对焦检测
    'focus_roi_size': 0.6,        # 关注图像中心区域的比例
    'adaptive_threshold': True,   # 使用自适应阈值
    'light_focus_mode': True,    # 轻量级对焦检测模式（更快但精度可能降低）
    'use_gpu_focus': True,        # 使用GPU加速对焦检测
    
    # 增量学习
    'update_frequency': 100,     # 每添加100张新照片更新一次模型
    
    # 设备配置
    'device': 'cuda',
    'mixed_precision': True,     # 使用混合精度训练
}

# 创建必要的目录
for dir_path in [CONFIG['output_dir'], CONFIG['models_dir']]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)