import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

from photo_classifier.config import CONFIG
from photo_classifier.utils.preprocessing.image_processor import ImageProcessor
from photo_classifier.utils.preprocessing.dataset import PhotoDataset, ContrastiveDataset, create_data_loaders, FocusDataset
from photo_classifier.models.simclr import SimCLR
from photo_classifier.utils.trainer import SimCLRTrainer
from photo_classifier.utils.clustering.cluster import PhotoClustering
from photo_classifier.utils.focus.focus_detection import FocusDetector

def parse_args():
    parser = argparse.ArgumentParser(description='无监督照片分类与对焦检测')
    parser.add_argument('--data_dir', type=str, default=CONFIG['data_dir'],
                        help='照片所在目录')
    parser.add_argument('--output_dir', type=str, default=CONFIG['output_dir'],
                        help='输出目录')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'incremental'], default='train',
                        help='运行模式: train(训练), predict(预测), incremental(增量学习)')
    parser.add_argument('--n_clusters', type=int, default=CONFIG['n_clusters'],
                        help='聚类数量')
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'],
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'],
                        help='训练轮数')
    parser.add_argument('--model_path', type=str, default='final_model.pth',
                        help='预训练模型路径')
    parser.add_argument('--cluster_model', type=str, default='cluster_model.pkl',
                        help='聚类模型路径')
    parser.add_argument('--optimize_clusters', action='store_true',
                        help='优化聚类数量')
    parser.add_argument('--focus_threshold', type=float, default=None,
                        help='自定义对焦阈值')
    parser.add_argument('--visualization', action='store_true',
                        help='生成可视化结果')
                        
    args = parser.parse_args()
    return args

def train_mode(args, config):
    """训练模式"""
    # 更新配置
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['n_clusters'] = args.n_clusters
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    
    if args.focus_threshold is not None:
        config['laplacian_threshold'] = args.focus_threshold
    
    print("配置信息:", config)
    
    # 初始化图像处理器
    processor = ImageProcessor(config)
    
    # 扫描图像
    print("扫描图像目录...")
    image_paths = processor.scan_image_directory()
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建数据加载器
    print("创建数据加载器...")
    contrastive_loader = create_data_loaders(config, image_paths, processor, contrastive=True)
    feature_loader = create_data_loaders(config, image_paths, processor, contrastive=False)
    
    # 创建检测对焦质量的数据集
    focus_dataset = FocusDataset(image_paths, processor)
    focus_loader = torch.utils.data.DataLoader(
        focus_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # 训练SimCLR模型
    trainer = SimCLRTrainer(config)
    
    # 如果指定了模型路径，加载模型
    if args.model_path and Path(config['models_dir']) / args.model_path.exists():
        print(f"加载预训练模型: {args.model_path}")
        trainer.load_model(args.model_path)
    else:
        print("从头开始训练模型...")
        trainer.train(contrastive_loader)
        
    # 提取特征
    print("提取图像特征...")
    features, image_paths = trainer.extract_features(feature_loader)
    
    # 获取对焦分数
    print("计算对焦分数...")
    focus_scores = []
    is_focused_list = []
    
    focus_detector = FocusDetector(
        laplacian_threshold=config['laplacian_threshold'],
        fft_threshold=config['fft_threshold']
    )
    
    for batch in tqdm(focus_loader, desc="Processing focus quality"):
        batch_scores = batch['focus_score'].numpy().flatten()
        focus_scores.extend(batch_scores)
        is_focused = batch['is_focused'].numpy().flatten()
        is_focused_list.extend(is_focused)
        
    focus_scores = np.array(focus_scores)
    is_focused_array = np.array(is_focused_list)
    
    print(f"对焦成功: {np.sum(is_focused_array)}/{len(is_focused_array)} ({np.mean(is_focused_array)*100:.2f}%)")
    
    # 聚类
    print("对特征进行聚类...")
    clusterer = PhotoClustering(config)
    
    # 如果指定了聚类模型，加载模型
    cluster_model_path = Path(config['models_dir']) / args.cluster_model
    if args.cluster_model and cluster_model_path.exists():
        print(f"加载聚类模型: {args.cluster_model}")
        clusterer.load(cluster_model_path)
        labels = clusterer.predict(features)
    else:
        # 如果需要优化聚类数量
        optimize = args.optimize_clusters
        labels = clusterer.fit(features, optimize=optimize)
        
        # 保存聚类模型
        clusterer.save(cluster_model_path)
        print(f"聚类模型已保存到: {cluster_model_path}")
    
    # 可视化特征
    if args.visualization:
        print("可视化特征分布...")
        trainer.visualize_features(features, labels, image_paths, focus_scores, method='tsne')
        trainer.visualize_features(features, labels, image_paths, focus_scores, method='pca')
    
    # 组织照片
    print("根据聚类结果组织照片...")
    organized_dir = clusterer.organize_photos(image_paths, labels, focus_scores)
    print(f"照片已按类别组织到: {organized_dir}")
    
    return features, labels, focus_scores, image_paths

def predict_mode(args, config):
    """预测模式 - 对新的照片进行分类"""
    # 更新配置
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    if args.focus_threshold is not None:
        config['laplacian_threshold'] = args.focus_threshold
    
    # 初始化图像处理器
    processor = ImageProcessor(config)
    
    # 扫描图像
    print("扫描图像目录...")
    image_paths = processor.scan_image_directory()
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建数据加载器
    print("创建数据加载器...")
    feature_loader = create_data_loaders(config, image_paths, processor, contrastive=False)
    
    # 创建对焦检测数据集
    focus_dataset = FocusDataset(image_paths, processor)
    focus_loader = torch.utils.data.DataLoader(
        focus_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # 加载训练好的SimCLR模型
    model_path = Path(config['models_dir']) / args.model_path
    if not model_path.exists():
        print(f"错误: 模型 {model_path} 不存在，请先训练模型")
        return
        
    trainer = SimCLRTrainer(config)
    trainer.load_model(args.model_path)
    
    # 提取特征
    print("提取图像特征...")
    features, image_paths = trainer.extract_features(feature_loader)
    
    # 获取对焦分数
    print("计算对焦分数...")
    focus_scores = []
    is_focused_list = []
    
    for batch in tqdm(focus_loader, desc="Processing focus quality"):
        batch_scores = batch['focus_score'].numpy().flatten()
        focus_scores.extend(batch_scores)
        is_focused = batch['is_focused'].numpy().flatten()
        is_focused_list.extend(is_focused)
        
    focus_scores = np.array(focus_scores)
    is_focused_array = np.array(is_focused_list)
    
    print(f"对焦成功: {np.sum(is_focused_array)}/{len(is_focused_array)} ({np.mean(is_focused_array)*100:.2f}%)")
    
    # 加载聚类模型
    cluster_model_path = Path(config['models_dir']) / args.cluster_model
    if not cluster_model_path.exists():
        print(f"错误: 聚类模型 {cluster_model_path} 不存在，请先训练聚类模型")
        return
        
    clusterer = PhotoClustering(config)
    clusterer.load(cluster_model_path)
    
    # 预测类别
    print("预测图片类别...")
    labels = clusterer.predict(features)
    
    # 可视化特征
    if args.visualization:
        print("可视化特征分布...")
        trainer.visualize_features(features, labels, image_paths, focus_scores, method='tsne')
        trainer.visualize_features(features, labels, image_paths, focus_scores, method='pca')
    
    # 组织照片
    print("根据聚类结果组织照片...")
    predict_output_dir = Path(config['output_dir']) / 'predictions'
    organized_dir = clusterer.organize_photos(image_paths, labels, focus_scores, predict_output_dir)
    print(f"照片已按类别组织到: {organized_dir}")
    
    return features, labels, focus_scores, image_paths

def incremental_mode(args, config):
    """增量学习模式 - 处理新添加的照片"""
    # 加载现有模型
    result = predict_mode(args, config)
    
    if result is None:
        print("增量学习失败: 无法加载现有模型")
        return
        
    features, labels, focus_scores, image_paths = result
    
    # 微调模型 (可选)
    if args.epochs > 0:
        print(f"使用新数据微调模型 ({args.epochs} epochs)...")
        
        # 初始化图像处理器
        processor = ImageProcessor(config)
        
        # 创建对比学习数据加载器
        contrastive_loader = create_data_loaders(config, image_paths, processor, contrastive=True)
        
        # 创建训练器并加载现有模型
        trainer = SimCLRTrainer(config)
        trainer.load_model(args.model_path)
        
        # 微调模型
        trainer.train(contrastive_loader, epochs=args.epochs)
        
        # 保存微调后的模型
        trainer.save_model('incremental_model.pth')
        
        # 重新提取特征
        feature_loader = create_data_loaders(config, image_paths, processor, contrastive=False)
        features, image_paths = trainer.extract_features(feature_loader)
        
        # 重新聚类
        print("重新聚类...")
        clusterer = PhotoClustering(config)
        if args.optimize_clusters:
            labels = clusterer.fit(features, optimize=True)
        else:
            clusterer.load(Path(config['models_dir']) / args.cluster_model)
            labels = clusterer.predict(features)
            
        # 保存更新的聚类模型
        clusterer.save(Path(config['models_dir']) / 'incremental_cluster.pkl')
        
        # 重新组织照片
        print("重新组织照片...")
        incremental_output_dir = Path(config['output_dir']) / 'incremental'
        organized_dir = clusterer.organize_photos(image_paths, labels, focus_scores, incremental_output_dir)
        print(f"照片已按类别组织到: {organized_dir}")
    
    return features, labels, focus_scores, image_paths

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if device.type == 'cpu':
        print("警告: 未检测到GPU，训练速度可能很慢")
        CONFIG['device'] = 'cpu'
        CONFIG['mixed_precision'] = False
    
    # 根据模式选择对应的功能
    if args.mode == 'train':
        train_mode(args, CONFIG)
    elif args.mode == 'predict':
        predict_mode(args, CONFIG)
    elif args.mode == 'incremental':
        incremental_mode(args, CONFIG)