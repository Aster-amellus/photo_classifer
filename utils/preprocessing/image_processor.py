import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import os
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.image_size = config['image_size']
        
        # 数据预处理转换
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # 数据增强转换（用于对比学习）
        self.augment = A.Compose([
            A.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.8, 1.0)),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def load_image(self, image_path):
        """加载图像文件"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def preprocess_image(self, image):
        """预处理单张图像"""
        return self.transform(image=image)["image"]
    
    def augment_image(self, image):
        """对图像进行数据增强，返回两个增强版本用于对比学习"""
        aug1 = self.augment(image=image)["image"]
        aug2 = self.augment(image=image)["image"]
        return aug1, aug2
    
    def scan_image_directory(self, data_dir=None):
        """扫描图像目录，返回所有图像路径"""
        if data_dir is None:
            data_dir = self.config['data_dir']
            
        data_dir = Path(data_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(data_dir.glob(f'**/*{ext}')))
            image_paths.extend(list(data_dir.glob(f'**/*{ext.upper()}')))
        
        return sorted(image_paths)
    
    def batch_preprocess_images(self, image_paths, output_dir=None, max_images=None):
        """批量预处理图像并保存"""
        if output_dir is None:
            output_dir = Path(self.config['output_dir']) / 'preprocessed'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        processed_paths = []
        
        for img_path in tqdm(image_paths, desc="Preprocessing images"):
            try:
                image = self.load_image(img_path)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                    
                processed = self.preprocess_image(image)
                
                # 生成输出文件名
                relative_path = img_path.relative_to(Path(self.config['data_dir']))
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存预处理后的图像
                processed_np = processed.permute(1, 2, 0).numpy() * 255
                processed_np = processed_np.astype(np.uint8)
                cv2.imwrite(str(output_path), cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR))
                
                processed_paths.append(output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return processed_paths