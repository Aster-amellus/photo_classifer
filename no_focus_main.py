#!/usr/bin/env python
"""
无对焦检测照片分类器 - 跳过对焦质量评估，直接进行聚类分类，适用于需要快速处理的情况
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# 导入必要的模块
from config import CONFIG
from utils.preprocessing.image_processor import ImageProcessor
from utils.preprocessing.dataset import PhotoDataset, ContrastiveDataset, create_data_loaders
from model.simclr import SimCLR
from utils.trainer import SimCLRTrainer
from utils.clustering.cluster import PhotoClustering

def parse_args():
    parser = argparse.ArgumentParser(description='无对焦检测的照片分类（更快速的处理）')
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
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--cluster_model', type=str, default=None,
                        help='聚类模型路径')
    parser.add_argument('--optimize_clusters', action='store_true',
                        help='优化聚类数量')                    
    return parser.parse_args()

def train_mode(args, config):
    """训练模式 - 无对焦检测"""
    # 更新配置
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['n_clusters'] = args.n_clusters
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    
    # 显式禁用对焦检测
    config['enable_focus_detection'] = False
    
    print("配置信息:", config)
    print("注意: 对焦检测已禁用，将直接进行分类")
    
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
    
    # 训练SimCLR模型
    trainer = SimCLRTrainer(config)
    
    # 如果指定了模型路径，加载模型
    if args.model_path:
        print(f"加载预训练模型: {args.model_path}")
        trainer.load_model(args.model_path)
    else:
        print("从头开始训练模型...")
        trainer.train(contrastive_loader)
        
    # 提取特征
    print("提取图像特征...")
    features, paths = trainer.extract_features(feature_loader)
    
    # 聚类
    print("对特征进行聚类...")
    clusterer = PhotoClustering(config)
    
    # 如果指定了聚类模型，加载模型
    if args.cluster_model and Path(args.cluster_model).exists():
        print(f"加载聚类模型: {args.cluster_model}")
        clusterer.load(args.cluster_model)
        labels = clusterer.predict(features)
    else:
        # 如果需要优化聚类数量
        optimize = args.optimize_clusters
        labels = clusterer.fit(features, optimize=optimize)
        
        # 保存聚类模型
        cluster_model_path = Path(config['models_dir']) / 'cluster_model.pkl'
        clusterer.save(cluster_model_path)
        print(f"聚类模型已保存到: {cluster_model_path}")
    
    # 可视化特征 - 不使用对焦分数
    print("可视化特征分布...")
    trainer.visualize_features(features, labels, paths, focus_scores=None, method='tsne')
    trainer.visualize_features(features, labels, paths, focus_scores=None, method='pca')
    
    # 组织照片 - 不使用对焦分数
    print("根据聚类结果组织照片...")
    # 创建虚拟对焦分数 - 全部设为100表示最佳对焦
    dummy_focus_scores = np.ones(len(features)) * 100
    organized_dir = clusterer.organize_photos(paths, labels, dummy_focus_scores)
    
    print(f"照片已组织到: {organized_dir}")
    print(f"模型训练和照片分类完成!")

def predict_mode(args, config):
    """预测模式 - 无对焦检测"""
    # 更新配置
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    # 显式禁用对焦检测
    config['enable_focus_detection'] = False
    
    # 检查模型路径
    if not args.model_path or not (Path(config['models_dir']) / args.model_path).exists():
        model_file = Path(config['models_dir']) / 'final_model.pth'
        if not model_file.exists():
            model_file = Path(config['models_dir']) / 'best_model.pth'
            
        if model_file.exists():
            args.model_path = model_file.name
        else:
            print("错误: 未找到模型文件，请提供有效的模型路径或先训练模型")
            return
            
    # 检查聚类模型
    if not args.cluster_model:
        cluster_model_file = Path(config['models_dir']) / 'cluster_model.pkl'
        if cluster_model_file.exists():
            args.cluster_model = str(cluster_model_file)
        else:
            print("错误: 未找到聚类模型，请提供有效的聚类模型路径或先训练模型")
            return
    
    # 初始化图像处理器
    processor = ImageProcessor(config)
    
    # 扫描图像
    print("扫描图像目录...")
    image_paths = processor.scan_image_directory()
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建数据加载器
    print("创建数据加载器...")
    feature_loader = create_data_loaders(config, image_paths, processor, contrastive=False)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    trainer = SimCLRTrainer(config)
    trainer.load_model(Path(config['models_dir']) / args.model_path)
    
    # 提取特征
    print("提取图像特征...")
    features, paths = trainer.extract_features(feature_loader)
    
    # 加载聚类模型
    print(f"加载聚类模型: {args.cluster_model}")
    clusterer = PhotoClustering(config)
    clusterer.load(args.cluster_model)
    
    # 预测类别
    labels = clusterer.predict(features)
    
    # 可视化特征 - 不使用对焦分数
    print("可视化特征分布...")
    trainer.visualize_features(features, labels, paths, focus_scores=None, method='tsne')
    
    # 组织照片 - 不使用对焦分数
    print("根据聚类结果组织照片...")
    # 创建虚拟对焦分数 - 全部设为100表示最佳对焦
    dummy_focus_scores = np.ones(len(features)) * 100  
    organized_dir = clusterer.organize_photos(paths, labels, dummy_focus_scores)
    
    print(f"照片已组织到: {organized_dir}")
    print("预测完成!")

def incremental_mode(args, config):
    """增量学习模式 - 处理新添加的照片 - 无对焦检测"""
    # 更新配置
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    # 显式禁用对焦检测
    config['enable_focus_detection'] = False
    
    # 检查模型路径
    if not args.model_path or not Path(config['models_dir']) / args.model_path.exists():
        model_file = Path(config['models_dir']) / 'final_model.pth'
        if not model_file.exists():
            model_file = Path(config['models_dir']) / 'best_model.pth'
            
        if model_file.exists():
            args.model_path = model_file.name
        else:
            print("错误: 未找到模型文件，请提供有效的模型路径或先训练模型")
            return
            
    # 检查聚类模型
    if not args.cluster_model:
        cluster_model_file = Path(config['models_dir']) / 'cluster_model.pkl'
        if cluster_model_file.exists():
            args.cluster_model = str(cluster_model_file)
        else:
            print("错误: 未找到聚类模型，请提供有效的聚类模型路径或先训练模型")
            return
    
    # 初始化图像处理器
    processor = ImageProcessor(config)
    
    # 检查已处理的图片记录
    processed_record = Path(config['output_dir']) / 'processed_images.txt'
    processed_images = set()
    
    if processed_record.exists():
        with open(processed_record, 'r') as f:
            processed_images = set(line.strip() for line in f.readlines())
    
    # 扫描图像
    print("扫描图像目录...")
    all_image_paths = processor.scan_image_directory()
    
    # 过滤出新图片
    new_image_paths = [p for p in all_image_paths if str(p) not in processed_images]
    print(f"找到 {len(new_image_paths)} 张新图片 (共 {len(all_image_paths)} 张)")
    
    if not new_image_paths:
        print("没有新图片需要处理!")
        return
        
    # 加载模型
    print(f"加载模型: {args.model_path}")
    trainer = SimCLRTrainer(config)
    trainer.load_model(args.model_path)
    
    # 创建数据加载器
    print("创建数据加载器...")
    feature_loader = create_data_loaders(config, new_image_paths, processor, contrastive=False)
    
    # 提取特征
    print("提取图像特征...")
    features, paths = trainer.extract_features(feature_loader)
    
    # 加载聚类模型
    print(f"加载聚类模型: {args.cluster_model}")
    clusterer = PhotoClustering(config)
    clusterer.load(args.cluster_model)
    
    # 预测类别
    labels = clusterer.predict(features)
    
    # 组织照片 - 不使用对焦分数
    print("根据聚类结果组织照片...")
    # 创建虚拟对焦分数 - 全部设为100表示最佳对焦
    dummy_focus_scores = np.ones(len(features)) * 100
    organized_dir = clusterer.organize_photos(paths, labels, dummy_focus_scores)
    
    # 记录处理过的图片
    with open(processed_record, 'a') as f:
        for path in paths:
            f.write(f"{path}\n")
    
    print(f"新照片已组织到: {organized_dir}")
    print("增量处理完成!")
    
    # 如果新照片数量超过阈值，考虑更新模型
    if len(new_image_paths) >= config['update_frequency']:
        print(f"新照片数量达到 {len(new_image_paths)} 张，考虑更新模型...")
        update = input("是否要使用新数据更新模型? (y/n): ")
        
        if update.lower() == 'y':
            # 创建对比学习数据加载器
            contrastive_loader = create_data_loaders(config, new_image_paths, processor, contrastive=True)
            
            # 微调模型
            print("使用新数据微调模型...")
            trainer.train(contrastive_loader, epochs=5)  # 使用较少的epoch进行微调
            
            # 保存更新后的模型
            model_path = Path(config['models_dir']) / f'updated_model_{time.strftime("%Y%m%d_%H%M%S")}.pth'
            trainer.save_model(model_path.name)
            print(f"更新后的模型已保存到: {model_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 使用命令行参数更新配置
    config = CONFIG.copy()
    
    # 设置输出目录
    if args.output_dir:
        config['output_dir'] = args.output_dir
        
    # 确保输出目录存在
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    print("注意: 此模式下不进行对焦检测，将直接进行照片分类")
    
    # 根据模式执行相应的功能
    if args.mode == 'train':
        train_mode(args, config)
    elif args.mode == 'predict':
        predict_mode(args, config)
    elif args.mode == 'incremental':
        incremental_mode(args, config)

if __name__ == "__main__":
    main()
