import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

from photo_classifier.utils.preprocessing.dataset import create_data_loaders
from photo_classifier.utils.focus.focus_detection import FocusDetector

class IncrementalLearner:
    """处理新增照片的增量学习类"""
    
    def __init__(self, config, trainer, clusterer, processor):
        self.config = config
        self.trainer = trainer
        self.clusterer = clusterer
        self.processor = processor
        self.focus_detector = FocusDetector(
            laplacian_threshold=config['laplacian_threshold'],
            fft_threshold=config['fft_threshold']
        )
        
        # 已处理图片记录文件
        self.processed_record = Path(config['output_dir']) / 'processed_images.txt'
        
    def load_processed_records(self):
        """加载已处理的图片记录"""
        processed_images = set()
        if self.processed_record.exists():
            with open(self.processed_record, 'r') as f:
                processed_images = set(line.strip() for line in f.readlines())
        return processed_images
    
    def save_processed_records(self, image_paths):
        """保存处理过的图片记录"""
        with open(self.processed_record, 'a') as f:
            for path in image_paths:
                f.write(f"{path}\n")
    
    def find_new_images(self, all_image_paths):
        """找出新添加的图片"""
        processed_images = self.load_processed_records()
        new_image_paths = [p for p in all_image_paths if str(p) not in processed_images]
        return new_image_paths
    
    def process_new_images(self, new_image_paths):
        """处理新增的图片"""
        if not new_image_paths:
            print("没有新图片需要处理!")
            return None, None, None
            
        # 创建数据加载器
        feature_loader = create_data_loaders(self.config, new_image_paths, self.processor, contrastive=False)
        
        # 提取特征
        features, paths = self.trainer.extract_features(feature_loader)
        
        # 计算对焦分数
        focus_scores = []
        for path in tqdm(paths, desc="Calculating focus scores"):
            img = self.processor.load_image(path)
            score = self.focus_detector.get_focus_score(img)
            focus_scores.append(score)
        
        focus_scores = np.array(focus_scores)
        
        # 预测类别
        labels = self.clusterer.predict(features)
        
        return features, labels, focus_scores
    
    def update_model(self, new_image_paths, epochs=5):
        """使用新图片更新模型"""
        # 创建对比学习数据加载器
        contrastive_loader = create_data_loaders(self.config, new_image_paths, self.processor, contrastive=True)
        
        # 微调模型
        self.trainer.train(contrastive_loader, epochs=epochs)
        
        # 保存更新后的模型
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = Path(self.config['models_dir']) / f'updated_model_{timestamp}.pth'
        self.trainer.save_model(model_path.name)
        
        return model_path
