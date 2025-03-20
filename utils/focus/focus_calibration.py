#!/usr/bin/env python
"""
对焦检测阈值校准工具 - 用于优化对焦检测算法的阈值参数
"""
import argparse
import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.focus.focus_detection import FocusDetector
from utils.focus.focus_utils import calibrate_focus_thresholds
from utils.preprocessing.image_processor import ImageProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='对焦检测参数校准工具')
    parser.add_argument('--focused_dir', type=str, required=True,
                        help='包含对焦良好图像的目录')
    parser.add_argument('--unfocused_dir', type=str, required=True,
                        help='包含对焦不佳图像的目录')
    parser.add_argument('--output', type=str, default='./focus_calibration_output',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='每类采样的图片数量，默认100')
    parser.add_argument('--image_size', type=int, default=512,
                        help='处理前调整图像大小')
    parser.add_argument('--adaptive', action='store_true',
                        help='是否启用自适应阈值')
    return parser.parse_args()

def load_sample_images(directory, num_samples, image_size=None):
    """从目录加载样本图像"""
    directory = Path(directory)
    
    # 查找所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(directory.glob(f'**/*{ext}')))
        image_paths.extend(list(directory.glob(f'**/*{ext.upper()}')))
    
    # 随机采样
    if len(image_paths) > num_samples:
        image_paths = random.sample(image_paths, num_samples)
    
    print(f"从 {directory} 加载 {len(image_paths)} 张图像")
    
    # 加载图像
    images = []
    for path in tqdm(image_paths, desc=f"加载图像"):
        img = cv2.imread(str(path))
        if img is None:
            print(f"警告: 无法加载图像 {path}")
            continue
            
        # 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        if image_size:
            img = cv2.resize(img, (image_size, image_size))
            
        images.append(img)
    
    return images

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载对焦良好的图像
    print(f"加载对焦良好的图像...")
    focused_images = load_sample_images(args.focused_dir, args.num_samples, args.image_size)
    
    # 加载对焦不佳的图像
    print(f"加载对焦不佳的图像...")
    unfocused_images = load_sample_images(args.unfocused_dir, args.num_samples, args.image_size)
    
    # 检查是否加载到了图像
    if not focused_images:
        print(f"错误: 未能从 {args.focused_dir} 加载到任何图像")
        sys.exit(1)
        
    if not unfocused_images:
        print(f"错误: 未能从 {args.unfocused_dir} 加载到任何图像")
        sys.exit(1)
    
    # 合并图像和标签
    all_images = focused_images + unfocused_images
    all_labels = np.array([1] * len(focused_images) + [0] * len(unfocused_images))
    
    # 创建默认检测器
    default_detector = FocusDetector(
        laplacian_threshold=100,
        fft_threshold=10,
        adaptive_threshold=args.adaptive,
        use_roi=True,
        focus_roi_size=0.6,
        use_weighted_regions=True
    )
    
    # 校准阈值
    print("开始校准对焦检测阈值...")
    calibration_results = calibrate_focus_thresholds(
        all_images, 
        all_labels, 
        default_detector
    )
    
    # 提取校准后的检测器
    calibrated_detector = calibration_results['detector']
    
    # 记录结果
    results = {
        'thresholds': calibration_results['thresholds'],
        'performance': calibration_results['performance'],
        'config': {
            'adaptive_threshold': args.adaptive,
            'focused_samples': len(focused_images),
            'unfocused_samples': len(unfocused_images),
            'image_size': args.image_size
        }
    }
    
    # 保存结果
    with open(output_dir / 'calibration_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 创建配置样本文件
    config_sample = f"""
# 校准的对焦检测配置
CONFIG = {{
    # ...其他配置...
    
    # 对焦检测参数 (已校准)
    'laplacian_threshold': {calibration_results['thresholds']['laplacian']:.2f},
    'fft_threshold': {calibration_results['thresholds']['fft']:.2f},
    'use_deep_focus': False,
    'focus_roi_size': 0.6,
    'adaptive_threshold': {str(args.adaptive).lower()},
    'use_weighted_regions': True,
    
    # ...其他配置...
}}
"""
    
    with open(output_dir / 'calibrated_config_sample.py', 'w') as f:
        f.write(config_sample)
    
    print(f"校准完成! 准确率: {results['performance']['accuracy']:.4f}, F1分数: {results['performance']['f1']:.4f}")
    print(f"已保存校准结果到: {output_dir}")
    print(f"校准后的阈值:")
    for name, value in results['thresholds'].items():
        print(f"  {name}: {value:.2f}")

if __name__ == "__main__":
    main()
