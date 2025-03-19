import cv2
import numpy as np
from skimage.filters import sobel

class FocusDetector:
    def __init__(self, laplacian_threshold=100, fft_threshold=10):
        self.laplacian_threshold = laplacian_threshold
        self.fft_threshold = fft_threshold
    
    def detect_focus_laplacian(self, image):
        """使用拉普拉斯算子方差判断图片是否对焦"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 计算拉普拉斯变换
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 计算方差
        score = np.var(laplacian)
        
        return score, score > self.laplacian_threshold
    
    def detect_focus_fft(self, image):
        """使用频域分析判断图片是否对焦"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
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
        
        return ratio, ratio > self.fft_threshold
    
    def detect_focus(self, image):
        """结合多种方法判断图片是否对焦"""
        laplacian_score, lap_focused = self.detect_focus_laplacian(image)
        fft_ratio, fft_focused = self.detect_focus_fft(image)
        
        # 结合两种方法的结果
        is_focused = lap_focused and fft_focused
        focus_metrics = {
            'laplacian_score': laplacian_score,
            'fft_ratio': fft_ratio,
            'is_focused': is_focused
        }
        
        return is_focused, focus_metrics
    
    def get_focus_score(self, image):
        """获取归一化的对焦分数(0-100)"""
        laplacian_score, _ = self.detect_focus_laplacian(image)
        fft_ratio, _ = self.detect_focus_fft(image)
        
        # 归一化拉普拉斯分数
        norm_lap = min(100, laplacian_score / self.laplacian_threshold * 100)
        
        # 归一化FFT比率
        norm_fft = min(100, fft_ratio / self.fft_threshold * 100)
        
        # 综合分数
        focus_score = 0.6 * norm_lap + 0.4 * norm_fft
        
        return focus_score