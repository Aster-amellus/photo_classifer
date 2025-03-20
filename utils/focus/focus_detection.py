import cv2
import numpy as np
from skimage.filters import sobel
import torch
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class FocusDetector:
    def __init__(self, laplacian_threshold=100, fft_threshold=10, 
                 use_roi=True, focus_roi_size=0.6, 
                 adaptive_threshold=False, use_weighted_regions=True,
                 use_gpu=True, light_mode=False, batch_size=16):
        self.laplacian_threshold = laplacian_threshold
        self.fft_threshold = fft_threshold
        self.use_roi = use_roi
        self.focus_roi_size = focus_roi_size
        self.adaptive_threshold = adaptive_threshold
        self.use_weighted_regions = use_weighted_regions
        self.light_mode = light_mode  # 轻量级模式：仅使用最快的检测方法
        self.batch_size = batch_size
        
        # GPU相关设置
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            print(f"对焦检测器将使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("对焦检测器将使用CPU")
            
        # 初始化GPU上的Sobel和Laplacian卷积核 - 适用于GPU加速
        if self.use_gpu:
            # Laplacian卷积核
            laplacian_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            self.laplacian_kernel = laplacian_kernel.expand(1, 1, 3, 3)
            
            # Sobel卷积核(x方向和y方向)
            sobel_x_kernel = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            
            sobel_y_kernel = torch.tensor([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            
            self.sobel_x_kernel = sobel_x_kernel.expand(1, 1, 3, 3)
            self.sobel_y_kernel = sobel_y_kernel.expand(1, 1, 3, 3)
        
        # 设置线程池
        max_workers = os.cpu_count() or 4
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    
    def get_roi(self, image):
        """获取图像的感兴趣区域（中心区域）"""
        h, w = image.shape[:2]
        roi_size = self.focus_roi_size
        
        # 计算ROI的坐标
        start_h = int(h * (1 - roi_size) / 2)
        end_h = int(h * (1 + roi_size) / 2)
        start_w = int(w * (1 - roi_size) / 2)
        end_w = int(w * (1 + roi_size) / 2)
        
        # 提取ROI
        roi = image[start_h:end_h, start_w:end_w]
        return roi
    
    def get_weighted_regions(self, image):
        """将图像分成九宫格并赋予不同权重分析对焦情况"""
        h, w = image.shape[:2]
        h_step, w_step = h // 3, w // 3
        
        # 使用九宫格权重：中心区域权重高，边缘区域权重低
        weights = np.array([
            [0.5, 0.8, 0.5],
            [0.8, 1.0, 0.8],
            [0.5, 0.8, 0.5]
        ])
        
        regions = []
        region_weights = []
        
        for i in range(3):
            for j in range(3):
                # 提取区域
                region = image[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                regions.append(region)
                region_weights.append(weights[i, j])
                
        return regions, region_weights
    
    def detect_focus_laplacian(self, image, use_roi=None):
        """使用拉普拉斯算子方差判断图片是否对焦"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if use_roi is None:
            use_roi = self.use_roi
            
        # 如果使用ROI，只分析中心区域
        if use_roi:
            gray = self.get_roi(gray)
            
        # 应用高斯模糊去除噪声
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
        # 计算拉普拉斯变换
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 计算方差
        score = np.var(laplacian)
        
        # 如果使用自适应阈值
        if self.adaptive_threshold:
            # 根据图像复杂度调整阈值
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            # 复杂图像需要更高的阈值
            adaptive_threshold = self.laplacian_threshold * (1 + edge_density * 2)
            is_focused = score > adaptive_threshold
        else:
            is_focused = score > self.laplacian_threshold
        
        return score, is_focused
    
    def detect_focus_laplacian_weighted(self, image):
        """使用加权区域的拉普拉斯算子判断对焦情况"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if self.use_weighted_regions:
            regions, weights = self.get_weighted_regions(gray)
            
            total_score = 0
            total_weight = sum(weights)
            
            for region, weight in zip(regions, weights):
                # 对每个区域计算拉普拉斯变换
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                region_score = np.var(laplacian)
                total_score += region_score * weight
                
            weighted_score = total_score / total_weight
            
            # 如果使用自适应阈值
            if self.adaptive_threshold:
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
                adaptive_threshold = self.laplacian_threshold * (1 + edge_density * 2)
                is_focused = weighted_score > adaptive_threshold
            else:
                is_focused = weighted_score > self.laplacian_threshold
                
            return weighted_score, is_focused
        else:
            # 如果不使用加权区域，使用标准拉普拉斯方法
            return self.detect_focus_laplacian(image)
    
    def detect_focus_fft(self, image, use_roi=None):
        """使用频域分析判断图片是否对焦"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if use_roi is None:
            use_roi = self.use_roi
            
        # 如果使用ROI，只分析中心区域
        if use_roi:
            gray = self.get_roi(gray)
            
        # 应用Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 计算频谱幅度
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 分离高频和低频部分
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建高频掩码（中心区域为低频）
        mask_low = np.zeros((rows, cols), np.uint8)
        mask_high = np.ones((rows, cols), np.uint8)
        
        r = min(rows, cols) // 8  # 低频区域半径
        cv2.circle(mask_low, (ccol, crow), r, 1, -1)
        cv2.circle(mask_high, (ccol, crow), r, 0, -1)
        
        # 计算高低频能量比
        high_energy = np.sum(magnitude_spectrum * mask_high)
        low_energy = np.sum(magnitude_spectrum * mask_low)
        
        ratio = high_energy / (low_energy + 1e-10)  # 避免除零
        
        # 自适应阈值
        if self.adaptive_threshold:
            # 根据图像尺寸和平均亮度调整阈值
            avg_brightness = np.mean(gray) / 255.0
            adaptive_threshold = self.fft_threshold * (0.8 + avg_brightness * 0.4)
            is_focused = ratio > adaptive_threshold
        else:
            is_focused = ratio > self.fft_threshold
        
        return ratio, is_focused
    
    def detect_focus_gradient(self, image, use_roi=None):
        """使用图像梯度判断对焦质量"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if use_roi is None:
            use_roi = self.use_roi
            
        # 如果使用ROI，只分析中心区域
        if use_roi:
            gray = self.get_roi(gray)
            
        # 计算Sobel梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅度
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 计算梯度的平均值和标准差
        mean_gradient = np.mean(gradient_magnitude)
        std_gradient = np.std(gradient_magnitude)
        
        # 计算得分：高梯度平均值和标准差表示图像清晰
        score = mean_gradient * std_gradient / 1000.0
        
        # 阈值（需要根据实际图像调整）
        threshold = 5.0
        if self.adaptive_threshold:
            # 根据图像亮度和对比度调整阈值
            avg_brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 128.0
            threshold = threshold * (0.7 + avg_brightness * 0.3) * (0.8 + contrast * 0.4)
            
        is_focused = score > threshold
        
        return score, is_focused
    
    def detect_focus_local_variance(self, image, use_roi=None, block_size=16):
        """使用局部方差分析对焦质量"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if use_roi is None:
            use_roi = self.use_roi
            
        # 如果使用ROI，只分析中心区域
        if use_roi:
            gray = self.get_roi(gray)
            
        h, w = gray.shape
        local_vars = []
        
        # 计算局部方差
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i + block_size, j:j + block_size]
                block_var = np.var(block)
                local_vars.append(block_var)
                
        # 使用排序后的局部方差分布
        sorted_vars = sorted(local_vars, reverse=True)
        
        # 使用前20%的局部方差的平均值评估对焦质量
        top_n = max(1, int(len(sorted_vars) * 0.2))
        score = np.mean(sorted_vars[:top_n])
        
        # 阈值
        threshold = 100.0
        if self.adaptive_threshold:
            avg_brightness = np.mean(gray) / 255.0
            threshold = threshold * (0.8 + avg_brightness * 0.4)
            
        is_focused = score > threshold
        
        return score, is_focused
    
    def detect_focus_dct(self, image, use_roi=None):
        """使用离散余弦变换(DCT)检测对焦质量"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if use_roi is None:
            use_roi = self.use_roi
            
        # 如果使用ROI，只分析中心区域
        if use_roi:
            gray = self.get_roi(gray)
            
        # 确保图像大小是8的倍数
        h, w = gray.shape
        h_pad = h if h % 8 == 0 else ((h // 8) + 1) * 8
        w_pad = w if w % 8 == 0 else ((w // 8) + 1) * 8
        
        # 填充图像
        gray_padded = np.zeros((h_pad, w_pad), dtype=np.float32)
        gray_padded[:h, :w] = gray
        
        # 分块处理DCT
        block_size = 8
        total_energy = 0
        ac_energy = 0
        
        for i in range(0, h_pad, block_size):
            for j in range(0, w_pad, block_size):
                block = gray_padded[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # 计算总能量和AC能量
                block_energy = np.sum(dct_block ** 2)
                dc_energy = dct_block[0, 0] ** 2
                block_ac_energy = block_energy - dc_energy
                
                total_energy += block_energy
                ac_energy += block_ac_energy
                
        # 计算AC/DC比率
        dc_energy = total_energy - ac_energy
        ratio = ac_energy / (dc_energy + 1e-10)
        
        # 阈值
        threshold = 0.5
        if self.adaptive_threshold:
            contrast = np.std(gray) / 128.0
            threshold = threshold * (0.8 + contrast * 0.4)
            
        is_focused = ratio > threshold
        
        return ratio, is_focused
    
    def detect_focus(self, image):
        """结合多种方法判断图片是否对焦"""
        # 拉普拉斯方法（加权）
        lap_score, lap_focused = self.detect_focus_laplacian_weighted(image)
        
        # FFT方法
        fft_ratio, fft_focused = self.detect_focus_fft(image)
        
        # 梯度方法
        gradient_score, gradient_focused = self.detect_focus_gradient(image)
        
        # 局部方差方法
        local_var_score, local_var_focused = self.detect_focus_local_variance(image)
        
        # DCT方法
        dct_ratio, dct_focused = self.detect_focus_dct(image)
        
        # 加权投票决定最终结果
        votes = [
            (lap_focused, 0.35),      # 拉普拉斯权重
            (fft_focused, 0.25),      # FFT权重
            (gradient_focused, 0.15),  # 梯度权重
            (local_var_focused, 0.15), # 局部方差权重
            (dct_focused, 0.10)        # DCT权重
        ]
        
        weighted_sum = sum(vote * weight for vote, weight in votes)
        is_focused = weighted_sum > 0.5  # 大于0.5表示加权多数认为对焦良好
        
        # 记录所有指标
        focus_metrics = {
            'laplacian_score': lap_score,
            'fft_ratio': fft_ratio,
            'gradient_score': gradient_score,
            'local_var_score': local_var_score,
            'dct_ratio': dct_ratio,
            'weighted_score': weighted_sum,
            'is_focused': is_focused
        }
        
        return is_focused, focus_metrics
    
    def get_focus_score(self, image):
        """获取归一化的对焦分数(0-100)"""
        # 获取所有指标
        is_focused, metrics = self.detect_focus(image)
        
        # 归一化分数
        # 1. 拉普拉斯分数
        norm_lap = min(100, metrics['laplacian_score'] / self.laplacian_threshold * 100)
        
        # 2. FFT比率
        norm_fft = min(100, metrics['fft_ratio'] / self.fft_threshold * 100)
        
        # 3. 梯度分数
        norm_gradient = min(100, metrics['gradient_score'] / 5.0 * 100)
        
        # 4. 局部方差分数
        norm_local_var = min(100, metrics['local_var_score'] / 100.0 * 100)
        
        # 5. DCT比率
        norm_dct = min(100, metrics['dct_ratio'] / 0.5 * 100)
        
        # 综合分数（加权平均）
        focus_score = (
            0.35 * norm_lap + 
            0.25 * norm_fft + 
            0.15 * norm_gradient + 
            0.15 * norm_local_var + 
            0.10 * norm_dct
        )
        
        return focus_score
    
    def analyze_focus_quality(self, image, return_visualization=False):
        """分析对焦质量并返回详细报告"""
        is_focused, metrics = self.detect_focus(image)
        focus_score = self.get_focus_score(image)
        
        # 创建结果字典
        result = {
            'is_focused': is_focused,
            'focus_score': focus_score,
            'metrics': metrics
        }
        
        # 如果需要可视化结果
        if return_visualization:
            vis_image = self._create_focus_visualization(image, is_focused, metrics)
            result['visualization'] = vis_image
            
        return result
    
    def _create_focus_visualization(self, image, is_focused, metrics):
        """创建对焦质量的可视化结果"""
        # 复制图像
        vis_image = image.copy()
        
        # 在图像上添加对焦状态指示
        color = (0, 255, 0) if is_focused else (255, 0, 0)  # 绿色表示对焦好，红色表示对焦差
        text = "Focused" if is_focused else "Unfocused"
        
        # 在图像上绘制文本和边框
        h, w = image.shape[:2]
        cv2.putText(vis_image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(vis_image, (10, 10), (w-10, h-10), color, 2)
        
        # 如果使用ROI，显示感兴趣区域
        if self.use_roi:
            roi_size = self.focus_roi_size
            start_h = int(h * (1 - roi_size) / 2)
            end_h = int(h * (1 + roi_size) / 2)
            start_w = int(w * (1 - roi_size) / 2)
            end_w = int(w * (1 + roi_size) / 2)
            
            cv2.rectangle(vis_image, (start_w, start_h), (end_w, end_h), (255, 255, 0), 2)
        
        # 添加指标信息
        metrics_text = [
            f"Score: {metrics['weighted_score']:.2f}",
            f"Lap: {metrics['laplacian_score']:.2f}",
            f"FFT: {metrics['fft_ratio']:.2f}",
            f"Grad: {metrics['gradient_score']:.2f}",
            f"Var: {metrics['local_var_score']:.2f}",
            f"DCT: {metrics['dct_ratio']:.2f}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(vis_image, text, (20, 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return vis_image
    
    def detect_focus_laplacian_gpu(self, image_tensor):
        """使用GPU加速的拉普拉斯算子检测对焦"""
        # 确保图像是灰度图并且在GPU上
        if len(image_tensor.shape) == 4 and image_tensor.shape[1] == 3:  # 批量RGB图像
            # [B, 3, H, W] -> [B, 1, H, W]
            image_tensor = 0.299 * image_tensor[:, 0] + 0.587 * image_tensor[:, 1] + 0.114 * image_tensor[:, 2]
            image_tensor = image_tensor.unsqueeze(1)
        elif len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:  # 单张RGB图像
            # [3, H, W] -> [1, H, W]
            image_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        
        # 应用Laplacian卷积
        laplacian = torch.nn.functional.conv2d(
            image_tensor, self.laplacian_kernel, padding=1
        )
        
        # 计算方差
        batch_var = torch.var(laplacian.view(laplacian.shape[0], -1), dim=1)
        
        # 判断是否对焦良好
        if self.adaptive_threshold:
            # 计算边缘密度 - 简化版本
            edge_intensity = torch.abs(laplacian).mean(dim=(1,2,3))
            adaptive_thresholds = self.laplacian_threshold * (1 + edge_intensity * 0.1)
            is_focused = batch_var > adaptive_thresholds
        else:
            is_focused = batch_var > self.laplacian_threshold
        
        return batch_var, is_focused
    
    def detect_focus_gradient_gpu(self, image_tensor):
        """使用GPU加速的梯度分析检测对焦"""
        # 确保图像是灰度图并且在GPU上
        if len(image_tensor.shape) == 4 and image_tensor.shape[1] == 3:  # 批量RGB图像
            # [B, 3, H, W] -> [B, 1, H, W]
            image_tensor = 0.299 * image_tensor[:, 0] + 0.587 * image_tensor[:, 1] + 0.114 * image_tensor[:, 2]
            image_tensor = image_tensor.unsqueeze(1)
        elif len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:  # 单张RGB图像
            # [3, H, W] -> [1, H, W]
            image_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        
        # 应用Sobel卷积
        grad_x = torch.nn.functional.conv2d(image_tensor, self.sobel_x_kernel, padding=1)
        grad_y = torch.nn.functional.conv2d(image_tensor, self.sobel_y_kernel, padding=1)
        
        # 计算梯度幅度
        grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-10)
        
        # 计算梯度的均值和标准差
        batch_mean = torch.mean(grad_magnitude.view(grad_magnitude.shape[0], -1), dim=1)
        batch_std = torch.std(grad_magnitude.view(grad_magnitude.shape[0], -1), dim=1)
        
        # 计算得分
        batch_scores = batch_mean * batch_std / 1000.0
        
        # 判断是否对焦良好
        threshold = 5.0
        if self.adaptive_threshold:
            brightness = torch.mean(image_tensor, dim=(1,2,3))
            thresholds = threshold * (0.7 + brightness * 0.3)
            is_focused = batch_scores > thresholds
        else:
            is_focused = batch_scores > threshold
        
        return batch_scores, is_focused
    
    def batch_detect_focus(self, images):
        """批量检测多张图像的对焦质量"""
        start_time = time.time()
        
        # 转换图像格式并移至GPU
        if isinstance(images, list):
            # 将多张图像转换为张量
            batch_tensor = self._prepare_batch_tensor(images)
        elif isinstance(images, np.ndarray):
            # 单张图像或已经是numpy数组的批量图像
            if len(images.shape) == 3:  # 单张图像
                images = [images]
                batch_tensor = self._prepare_batch_tensor(images)
            else:  # 批量图像
                batch_tensor = self._prepare_batch_tensor(list(images))
        else:
            raise ValueError("不支持的输入类型，请提供图像列表或numpy数组")
        
        # 获取批量大小
        batch_size = batch_tensor.shape[0]
        
        # 提取ROI区域
        if self.use_roi:
            batch_tensor = self._extract_batch_roi(batch_tensor)
        
        # 快速模式：仅使用最快的检测方法
        if self.light_mode:
            # 使用GPU加速的拉普拉斯算子方法
            scores, is_focused = self.detect_focus_laplacian_gpu(batch_tensor)
            
            # 转换为CPU并转为numpy
            scores = scores.cpu().numpy()
            is_focused = is_focused.cpu().numpy()
            
            # 构建结果
            results = []
            for i in range(batch_size):
                result = {
                    'is_focused': bool(is_focused[i]),
                    'focus_score': float(scores[i] / self.laplacian_threshold * 100),
                    'metrics': {
                        'laplacian_score': float(scores[i])
                    }
                }
                results.append(result)
        else:
            # 完整模式：使用所有方法并加权评估
            # GPU加速的拉普拉斯算子方法
            lap_scores, lap_focused = self.detect_focus_laplacian_gpu(batch_tensor)
            
            # GPU加速的梯度方法
            grad_scores, grad_focused = self.detect_focus_gradient_gpu(batch_tensor)
            
            # 将结果转移到CPU
            lap_scores = lap_scores.cpu().numpy()
            lap_focused = lap_focused.cpu().numpy()
            grad_scores = grad_scores.cpu().numpy()
            grad_focused = grad_focused.cpu().numpy()
            
            # 加权评估
            results = []
            for i in range(batch_size):
                # 加权投票
                weighted_score = lap_focused[i] * 0.6 + grad_focused[i] * 0.4
                is_focused = weighted_score > 0.5
                
                # 计算总分
                focus_score = (
                    0.6 * min(100, lap_scores[i] / self.laplacian_threshold * 100) + 
                    0.4 * min(100, grad_scores[i] / 5.0 * 100)
                )
                
                result = {
                    'is_focused': bool(is_focused),
                    'focus_score': float(focus_score),
                    'metrics': {
                        'laplacian_score': float(lap_scores[i]),
                        'gradient_score': float(grad_scores[i]),
                        'weighted_score': float(weighted_score)
                    }
                }
                results.append(result)
        
        elapsed = time.time() - start_time
        if len(results) > 1:
            print(f"批量处理 {len(results)} 张图像耗时: {elapsed:.3f}秒 (每张 {elapsed/len(results)*1000:.1f}毫秒)")
        
        return results if len(results) > 1 else results[0]
    
    def _prepare_batch_tensor(self, images):
        """将图像列表转换为批量张量并移至GPU"""
        batch_tensor = []
        
        for img in images:
            # 转换为RGB格式（如果不是）
            if len(img.shape) == 2:  # 灰度图
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3 and img.dtype == np.uint8:  # RGB图像但是uint8格式
                img = img.astype(np.float32) / 255.0
            
            # 转换为PyTorch张量格式 [C, H, W]
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            batch_tensor.append(img_tensor)
        
        # 堆叠为批量张量 [B, C, H, W]
        batch_tensor = torch.stack(batch_tensor)
        
        # 移至GPU
        if self.use_gpu:
            batch_tensor = batch_tensor.to(self.device)
        
        return batch_tensor
    
    def _extract_batch_roi(self, batch_tensor):
        """提取批量张量中每个图像的ROI区域"""
        B, C, H, W = batch_tensor.shape
        roi_size = self.focus_roi_size
        
        # 计算ROI坐标
        start_h = int(H * (1 - roi_size) / 2)
        end_h = int(H * (1 + roi_size) / 2)
        start_w = int(W * (1 - roi_size) / 2)
        end_w = int(W * (1 + roi_size) / 2)
        
        # 提取ROI区域
        roi_tensor = batch_tensor[:, :, start_h:end_h, start_w:end_w]
        
        return roi_tensor
    
    def analyze_focus_quality(self, image, return_visualization=False):
        """分析单张图像的对焦质量"""
        # 使用新的批量处理接口
        result = self.batch_detect_focus([image])
        
        # 如果需要可视化
        if return_visualization:
            vis_image = self._create_focus_visualization(image, result['is_focused'], result['metrics'])
            result['visualization'] = vis_image
        
        return result
    
    def analyze_images_batch(self, images, return_visualizations=False):
        """批量分析多张图像的对焦质量"""
        # 使用新的批量处理接口处理所有图像
        results = self.batch_detect_focus(images)
        
        # 如果需要可视化
        if return_visualizations:
            visualizations = []
            for i, result in enumerate(results):
                vis_image = self._create_focus_visualization(
                    images[i], result['is_focused'], result['metrics']
                )
                results[i]['visualization'] = vis_image
                visualizations.append(vis_image)
            
            return results, visualizations
        
        return results