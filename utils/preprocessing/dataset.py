import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import numpy as np
from photo_classifier.utils.focus.focus_detection import FocusDetector

class PhotoDataset(Dataset):
    def __init__(self, image_paths, processor, transform=None, return_path=False):
        self.image_paths = image_paths
        self.processor = processor
        self.transform = transform
        self.return_path = return_path
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.processor.load_image(image_path)
        
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = self.processor.preprocess_image(image)
        
        if self.return_path:
            return image, str(image_path)
        return image

class ContrastiveDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.processor.load_image(image_path)
        
        # 生成两个增强版本
        aug1, aug2 = self.processor.augment_image(image)
        
        return aug1, aug2, str(image_path)

def create_data_loaders(config, image_paths, processor, contrastive=False):
    """创建数据加载器"""
    if contrastive:
        dataset = ContrastiveDataset(image_paths, processor)
    else:
        dataset = PhotoDataset(image_paths, processor, return_path=True)
        
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True if contrastive else False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return loader

class FocusDataset(Dataset):
    """用于对焦检测的数据集"""
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.focus_detector = FocusDetector()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.processor.load_image(image_path)
        
        # 计算对焦分数
        is_focused, metrics = self.focus_detector.detect_focus(image)
        focus_score = self.focus_detector.get_focus_score(image)
        
        # 预处理图像
        processed_image = self.processor.preprocess_image(image)
        
        return {
            'image': processed_image,
            'path': str(image_path),
            'is_focused': torch.tensor([1.0 if is_focused else 0.0]),
            'focus_score': torch.tensor([focus_score]),
            'laplacian_score': torch.tensor([metrics['laplacian_score']]),
            'fft_ratio': torch.tensor([metrics['fft_ratio']])
        }