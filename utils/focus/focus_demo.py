#!/usr/bin/env python
"""
对焦检测演示程序 - 用于测试和展示对焦检测功能
"""
import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.focus.focus_detection import FocusDetector
from utils.focus.focus_utils import compare_focus_methods, analyze_focus_distribution, batch_analyze_images
from utils.preprocessing.image_processor import ImageProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='对焦检测演示程序')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录路径')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'compare'], default='single', help='运行模式: single(单张图像), batch(批处理), compare(比较方法)')
    parser.add_argument('--lap_threshold', type=float, default=100.0, help='拉普拉斯阈值')
    parser.add_argument('--fft_threshold', type=float, default=10.0, help='FFT阈值')
    parser.add_argument('--roi_size', type=float, default=0.6, help='ROI大小比例')
    parser.add_argument('--adaptive', action='store_true', help='是否使用自适应阈值')
    parser.add_argument('--image_size', type=int, default=384, help='图像大小')
    return parser.parse_args()

def process_single_image(image_path, detector, output_dir, image_size):
    """处理单张图像"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 调整图像大小
    image = cv2.resize(image, (image_size, image_size))
    
    # 分析对焦质量
    result = detector.analyze_focus_quality(image, return_visualization=True)
    
    # 保存结果
    output_path = output_dir / f"{Path(image_path).stem}_analysis.jpg"
    cv2.imwrite(str(output_path), result['visualization'])
    print(f"结果已保存到: {output_path}")

def batch_process_images(input_dir, detector, output_dir, image_size):
    """批量处理图像"""
    image_paths = list(Path(input_dir).glob('*.jpg'))
    processor = ImageProcessor({'image_size': image_size})
    batch_analyze_images(image_paths, processor, detector, output_dir)

def compare_methods(input_dir, detector, output_dir, image_size):
    """比较不同方法的对焦检测结果"""
    image_paths = list(Path(input_dir).glob('*.jpg'))
    for image_path in tqdm(image_paths, desc="Comparing methods"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法加载图像: {image_path}")
            continue
        
        # 调整图像大小
        image = cv2.resize(image, (image_size, image_size))
        
        # 比较方法
        comparison_image = compare_focus_methods(image, detector)
        
        # 保存结果
        output_path = output_dir / f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(str(output_path), comparison_image)
        print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    # 配置命令行参数解析
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建对焦检测器
    detector = FocusDetector(
        laplacian_threshold=args.lap_threshold,
        fft_threshold=args.fft_threshold,
        use_roi=True,
        focus_roi_size=args.roi_size,
        adaptive_threshold=args.adaptive,
        use_weighted_regions=True
    )
    
    # 处理输入路径
    input_path = Path(args.input)
    
    # 根据模式处理
    if args.mode == 'single':
        if input_path.is_file():
            process_single_image(str(input_path), detector, output_dir, args.image_size)
        else:
            print(f"错误: 单张图像模式需要输入一个图像文件，而不是目录: {input_path}")
            sys.exit(1)
            
    elif args.mode == 'batch':
        if input_path.is_dir():
            batch_process_images(input_path, detector, output_dir, args.image_size)
        else:
            print(f"错误: 批处理模式需要输入一个目录，而不是单个文件: {input_path}")
            sys.exit(1)
            
    elif args.mode == 'compare':
        if input_path.is_dir():
            compare_methods(input_path, detector, output_dir, args.image_size)
        else:
            print(f"错误: 比较模式需要输入一个目录，而不是单个文件: {input_path}")
            sys.exit(1)
    
    print(f"完成! 结果已保存到: {output_dir}")