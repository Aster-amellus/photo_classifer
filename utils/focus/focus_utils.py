import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
import seaborn as sns

def compare_focus_methods(image, detector):
    """比较不同对焦检测方法的结果"""
    # 获取原始图像尺寸
    h, w = image.shape[:2]
    
    # 1. 拉普拉斯方法
    lap_score, lap_focused = detector.detect_focus_laplacian(image)
    
    # 2. FFT方法
    fft_ratio, fft_focused = detector.detect_focus_fft(image)
    
    # 3. 梯度方法
    gradient_score, gradient_focused = detector.detect_focus_gradient(image)
    
    # 4. 局部方差方法
    local_var_score, local_var_focused = detector.detect_focus_local_variance(image)
    
    # 5. DCT方法
    dct_ratio, dct_focused = detector.detect_focus_dct(image)
    
    # 6. 组合方法
    is_focused, metrics = detector.detect_focus(image)
    focus_score = detector.get_focus_score(image)
    
    # 创建可视化结果
    # 转换为BGR格式，用于OpenCV绘图
    if image.shape[2] == 3:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 创建拉普拉斯可视化
    lap_image = bgr_image.copy()
    lap_color = (0, 255, 0) if lap_focused else (0, 0, 255)
    cv2.putText(lap_image, f"Laplacian: {lap_score:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, lap_color, 2)
    
    # 创建FFT可视化
    fft_image = bgr_image.copy()
    fft_color = (0, 255, 0) if fft_focused else (0, 0, 255)
    cv2.putText(fft_image, f"FFT: {fft_ratio:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fft_color, 2)
    
    # 创建梯度可视化
    grad_image = bgr_image.copy()
    grad_color = (0, 255, 0) if gradient_focused else (0, 0, 255)
    cv2.putText(grad_image, f"Gradient: {gradient_score:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, grad_color, 2)
    
    # 创建局部方差可视化
    var_image = bgr_image.copy()
    var_color = (0, 255, 0) if local_var_focused else (0, 0, 255)
    cv2.putText(var_image, f"Variance: {local_var_score:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, var_color, 2)
    
    # 创建DCT可视化
    dct_image = bgr_image.copy()
    dct_color = (0, 255, 0) if dct_focused else (0, 0, 255)
    cv2.putText(dct_image, f"DCT: {dct_ratio:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, dct_color, 2)
    
    # 创建组合方法可视化
    combined_image = bgr_image.copy()
    combined_color = (0, 255, 0) if is_focused else (0, 0, 255)
    cv2.putText(combined_image, f"Combined: {focus_score:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, combined_color, 2)
    
    # 创建完整图像网格
    grid_h, grid_w = h * 2, w * 3
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 前排
    grid_image[0:h, 0:w] = lap_image
    grid_image[0:h, w:2*w] = fft_image
    grid_image[0:h, 2*w:3*w] = grad_image
    
    # 后排
    grid_image[h:2*h, 0:w] = var_image
    grid_image[h:2*h, w:2*w] = dct_image
    grid_image[h:2*h, 2*w:3*w] = combined_image
    
    return grid_image

def analyze_focus_distribution(images, detector, output_dir=None):
    """分析一组图像的对焦质量分布"""
    focus_scores = []
    is_focused_list = []
    
    # 计算每张图像的对焦分数
    for image in tqdm(images, desc="Analyzing focus quality"):
        is_focused, metrics = detector.detect_focus(image)
        focus_score = detector.get_focus_score(image)
        
        focus_scores.append(focus_score)
        is_focused_list.append(1 if is_focused else 0)
        
    focus_scores = np.array(focus_scores)
    is_focused_array = np.array(is_focused_list)
    
    # 统计信息
    num_focused = np.sum(is_focused_array)
    percent_focused = num_focused / len(is_focused_array) * 100
    avg_score = np.mean(focus_scores)
    median_score = np.median(focus_scores)
    
    print(f"对焦统计:")
    print(f"  总图像数: {len(images)}")
    print(f"  对焦良好: {num_focused} ({percent_focused:.2f}%)")
    print(f"  平均分数: {avg_score:.2f}")
    print(f"  中位数分数: {median_score:.2f}")
    
    # 绘制分布图
    plt.figure(figsize=(12, 6))
    
    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(focus_scores, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=50, color='r', linestyle='--', label='阈值')
    plt.xlabel('对焦分数')
    plt.ylabel('图像数量')
    plt.title(f'对焦分数分布 (平均: {avg_score:.2f})')
    plt.legend()
    
    # 饼图
    plt.subplot(1, 2, 2)
    plt.pie([num_focused, len(images) - num_focused], 
            labels=['对焦良好', '对焦不佳'], 
            autopct='%1.1f%%',
            colors=['green', 'red'])
    plt.title('对焦质量分类')
    
    plt.tight_layout()
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'focus_distribution.png', dpi=300)
    
    plt.close()
    
    return {
        'focus_scores': focus_scores,
        'is_focused': is_focused_array,
        'stats': {
            'total': len(images),
            'focused': num_focused,
            'percent_focused': percent_focused,
            'avg_score': avg_score,
            'median_score': median_score
        }
    }

def calibrate_focus_thresholds(images, ground_truth, initial_detector=None):
    """自动校准对焦检测阈值"""
    # 如果没有提供检测器，创建一个默认的
    if initial_detector is None:
        initial_detector = FocusDetector(
            laplacian_threshold=100,
            fft_threshold=10,
            adaptive_threshold=False
        )
    
    # 收集各种方法的分数
    lap_scores = []
    fft_ratios = []
    gradient_scores = []
    local_var_scores = []
    dct_ratios = []
    
    print("收集特征分数...")
    for image in tqdm(images):
        # 拉普拉斯
        lap_score, _ = initial_detector.detect_focus_laplacian(image)
        lap_scores.append(lap_score)
        
        # FFT
        fft_ratio, _ = initial_detector.detect_focus_fft(image)
        fft_ratios.append(fft_ratio)
        
        # 梯度
        gradient_score, _ = initial_detector.detect_focus_gradient(image)
        gradient_scores.append(gradient_score)
        
        # 局部方差
        local_var_score, _ = initial_detector.detect_focus_local_variance(image)
        local_var_scores.append(local_var_score)
        
        # DCT
        dct_ratio, _ = initial_detector.detect_focus_dct(image)
        dct_ratios.append(dct_ratio)
    
    # 转换为NumPy数组
    lap_scores = np.array(lap_scores)
    fft_ratios = np.array(fft_ratios)
    gradient_scores = np.array(gradient_scores)
    local_var_scores = np.array(local_var_scores)
    dct_ratios = np.array(dct_ratios)
    
    # 查找每种方法的最佳阈值
    print("计算最佳阈值...")
    
    # 拉普拉斯
    precision, recall, thresholds = precision_recall_curve(ground_truth, lap_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_lap_threshold = thresholds[np.argmax(f1_scores)]
    
    # FFT
    precision, recall, thresholds = precision_recall_curve(ground_truth, fft_ratios)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_fft_threshold = thresholds[np.argmax(f1_scores)]
    
    # 梯度
    precision, recall, thresholds = precision_recall_curve(ground_truth, gradient_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_gradient_threshold = thresholds[np.argmax(f1_scores)]
    
    # 局部方差
    precision, recall, thresholds = precision_recall_curve(ground_truth, local_var_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_local_var_threshold = thresholds[np.argmax(f1_scores)]
    
    # DCT
    precision, recall, thresholds = precision_recall_curve(ground_truth, dct_ratios)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_dct_threshold = thresholds[np.argmax(f1_scores)]
    
    print(f"最佳阈值:")
    print(f"  拉普拉斯: {best_lap_threshold:.2f}")
    print(f"  FFT: {best_fft_threshold:.2f}")
    print(f"  梯度: {best_gradient_threshold:.2f}")
    print(f"  局部方差: {best_local_var_threshold:.2f}")
    print(f"  DCT: {best_dct_threshold:.2f}")
    
    # 创建校准后的检测器
    calibrated_detector = FocusDetector(
        laplacian_threshold=best_lap_threshold,
        fft_threshold=best_fft_threshold,
        adaptive_threshold=initial_detector.adaptive_threshold,
        use_roi=initial_detector.use_roi,
        focus_roi_size=initial_detector.focus_roi_size,
        use_weighted_regions=initial_detector.use_weighted_regions
    )
    
    # 评估校准后的检测器
    predictions = []
    for image in tqdm(images, desc="评估校准后的检测器"):
        is_focused, _ = calibrated_detector.detect_focus(image)
        predictions.append(1 if is_focused else 0)
    
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == ground_truth)
    f1 = f1_score(ground_truth, predictions)
    
    print(f"校准后性能:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  F1分数: {f1:.4f}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(ground_truth, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['模糊', '清晰'],
                yticklabels=['模糊', '清晰'])
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.title('混淆矩阵')
    
    # 保存结果
    if output_dir:
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    
    plt.close()
    
    return {
        'detector': calibrated_detector,
        'thresholds': {
            'laplacian': best_lap_threshold,
            'fft': best_fft_threshold,
            'gradient': best_gradient_threshold,
            'local_var': best_local_var_threshold,
            'dct': best_dct_threshold
        },
        'performance': {
            'accuracy': accuracy,
            'f1': f1
        }
    }

def batch_analyze_images(image_paths, processor, detector, output_dir=None):
    """批量分析图像的对焦质量并保存结果"""
    results = []
    
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        # 创建目录
        (output_dir / 'focused').mkdir(parents=True, exist_ok=True)
        (output_dir / 'unfocused').mkdir(parents=True, exist_ok=True)
        (output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
    
    for i, path in enumerate(tqdm(image_paths, desc="分析图像对焦质量")):
        try:
            # 加载图像
            image = processor.load_image(path)
            if image is None:
                print(f"无法加载图像: {path}")
                continue
                
            # 分析对焦质量
            result = detector.analyze_focus_quality(image, return_visualization=bool(output_dir))
            
            # 记录结果
            result['path'] = str(path)
            result['filename'] = Path(path).name
            results.append(result)
            
            # 保存结果
            if output_dir:
                # 根据对焦质量选择目录
                subdir = 'focused' if result['is_focused'] else 'unfocused'
                dest_path = output_dir / subdir / Path(path).name
                
                # 复制图像
                cv2.imwrite(str(dest_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # 保存可视化
                if 'visualization' in result:
                    vis_path = output_dir / 'visualizations' / f"{Path(path).stem}_analysis.jpg"
                    cv2.imwrite(str(vis_path), result['visualization'])
        
        except Exception as e:
            print(f"处理图像 {path} 时出错: {e}")
    
    # 创建摘要报告
    if output_dir:
        # 提取统计信息
        total = len(results)
        focused = sum(1 for r in results if r['is_focused'])
        percent_focused = focused / total * 100 if total > 0 else 0
        
        # 平均分数
        avg_score = np.mean([r['focus_score'] for r in results])
        
        # 创建分数分布图
        plt.figure(figsize=(10, 6))
        scores = [r['focus_score'] for r in results]
        plt.hist(scores, bins=20, alpha=0.7, color='blue')
        plt.axvline(x=50, color='r', linestyle='--', label='阈值')
        plt.xlabel('对焦分数')
        plt.ylabel('图像数量')
        plt.title(f'对焦分数分布 (平均: {avg_score:.2f})')
        plt.legend()
        plt.savefig(output_dir / 'score_distribution.png', dpi=300)
        plt.close()
        
        # 创建CSV报告
        import csv
        with open(output_dir / 'focus_analysis.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Focus Score', 'Is Focused', 'Laplacian Score', 'FFT Ratio', 
                             'Gradient Score', 'Local Variance Score', 'DCT Ratio'])
            
            for r in results:
                writer.writerow([
                    r['filename'],
                    f"{r['focus_score']:.2f}",
                    'Yes' if r['is_focused'] else 'No',
                    f"{r['metrics']['laplacian_score']:.2f}",
                    f"{r['metrics']['fft_ratio']:.2f}",
                    f"{r['metrics']['gradient_score']:.2f}",
                    f"{r['metrics']['local_var_score']:.2f}",
                    f"{r['metrics']['dct_ratio']:.2f}"
                ])
        
        # 创建摘要文本报告
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write(f"对焦质量分析摘要\n")
            f.write(f"----------------\n")
            f.write(f"分析的图像总数: {total}\n")
            f.write(f"对焦良好的图像: {focused} ({percent_focused:.2f}%)\n")
            f.write(f"对焦不佳的图像: {total - focused} ({100 - percent_focused:.2f}%)\n")
            f.write(f"平均对焦分数: {avg_score:.2f}\n")
    
    return results
