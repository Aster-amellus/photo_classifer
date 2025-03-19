# 无监督照片分类与对焦检测系统

这是一个基于无监督学习的系统，可以自动对日常照片进行分类并检测对焦质量。该系统使用对比学习和聚类算法，无需手动标注就能发现照片中的自然分组，并能检测照片的对焦质量。

## 系统特点

- **无监督照片分类**：使用SimCLR对比学习框架 + 混合聚类算法自动发现照片类别
- **自动对焦检测**：基于拉普拉斯变换和频域分析的双重对焦质量评估
- **高效处理**：针对NVIDIA RTX 4060优化，支持混合精度训练
- **增量学习**：能够处理新增的照片，无需完全重新训练
- **可视化分析**：提供聚类结果和特征分布的直观可视化
- **本地运行**：所有处理在本地完成，保护隐私

## 系统要求

- Python 3.8 或更高版本
- NVIDIA GPU (推荐RTX 4060或更高)
- CUDA 11.7 或更高版本
- 至少8GB RAM (推荐16GB或更高)
- 足够的硬盘空间用于存储照片和模型

## 安装指南

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/photo-classifier.git
cd photo-classifier

# 创建虚拟环境 (推荐)
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证CUDA安装

```python
# 运行以下Python代码验证CUDA是否正确安装
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"当前GPU: {torch.cuda.get_device_name(0)}")
```

### 3. 准备数据

将您想要分类的照片放在 `data/photos` 目录中。您可以使用子目录组织原始照片，系统会递归地扫描所有照片。

## 使用指南

### 基本用法

系统有三种主要运行模式：

1. **训练模式**：训练新模型并对照片进行分类
2. **预测模式**：使用已有模型对新照片进行分类
3. **增量模式**：处理新添加的照片，可选择微调现有模型

### 训练新模型

```bash
# Linux/Mac:
./run.sh --mode train --data_dir path/to/your/photos --optimize_clusters

# Windows:
.\run.ps1 -mode train -data_dir path/to/your/photos -optimize_clusters
```

主要参数：
- `--data_dir`: 照片所在目录路径
- `--output_dir`: 输出目录 (默认为 "output")
- `--optimize_clusters`: 自动优化聚类数量
- `--n_clusters`: 手动指定聚类数量 (默认为20)
- `--batch_size`: 批处理大小 (默认为64)
- `--epochs`: 训练轮数 (默认为20)
- `--focus_threshold`: 自定义对焦阈值

### 对新照片进行分类

```bash
# Linux/Mac:
./run.sh --mode predict --data_dir path/to/new/photos --model_path final_model.pth --cluster_model cluster_model.pkl

# Windows:
.\run.ps1 -mode predict -data_dir path/to/new/photos -model_path final_model.pth -cluster_model cluster_model.pkl
```

主要参数：
- `--data_dir`: 新照片所在目录
- `--model_path`: 预训练模型路径 (默认为 "final_model.pth")
- `--cluster_model`: 聚类模型路径 (默认为 "cluster_model.pkl")

### 增量学习

```bash
# Linux/Mac:
./run.sh --mode incremental --data_dir path/to/new/photos --epochs 5

# Windows:
.\run.ps1 -mode incremental -data_dir path/to/new/photos -epochs 5
```

主要参数：
- `--data_dir`: 新照片所在目录
- `--epochs`: 微调轮数 (设为0则不进行微调)

### 其他有用选项

- `--visualization`: 生成可视化结果
- `--focus_threshold`: 自定义对焦阈值，调整对焦检测灵敏度

## 输出说明

系统的输出包括：

1. **组织后的照片**：按照聚类结果组织在输出目录下
   - 每个聚类有自己的目录 (`cluster_0`, `cluster_1`等)
   - 对焦不良的照片会被单独放在每个聚类的`low_focus`子目录中

2. **可视化结果**：
   - 聚类评估指标图表
   - t-SNE和PCA特征可视化
   - 交互式HTML可视化 (需安装plotly)

3. **模型和日志**：
   - 训练好的模型保存在 `models` 目录
   - TensorBoard日志保存在 `output/logs` 目录

## 系统工作流程

1. **特征提取**：使用预训练的ResNet50模型从照片中提取视觉特征
2. **对比学习**：通过SimCLR框架优化特征提取，使语义相似的照片在特征空间中更接近
3. **对焦检测**：使用拉普拉斯变换和频域分析评估照片对焦质量
4. **聚类分析**：结合K-means和DBSCAN对特征进行聚类，自动发现照片的自然分组
5. **照片组织**：根据聚类结果和对焦质量将照片组织到相应目录

## 调优建议

- **聚类数量**：默认的聚类数量可能不适合所有照片集合，使用`--optimize_clusters`选项让系统自动寻找最佳聚类数量
- **对焦阈值**：如果对焦检测过于严格或宽松，可使用`--focus_threshold`调整阈值
- **批处理大小**：如果遇到内存不足错误，尝试减小批处理大小 (`--batch_size 32`或更小)
- **训练轮数**：对于大型照片集合，可以增加训练轮数 (`--epochs 30`或更多)以获得更好的特征表示

## 常见问题解决

1. **内存不足错误**：
   - 减小批处理大小 (`--batch_size 32`或更小)
   - 减小图像尺寸 (修改`config.py`中的`image_size`)

2. **GPU显存不足**：
   - 确保使用混合精度训练 (默认开启)
   - 减小批处理大小
   - 使用更小的backbone网络 (修改`config.py`中的`pretrained_model`为"resnet18")

3. **训练速度慢**：
   - 确保正在使用GPU (检查输出中的设备信息)
   - 增加工作线程数 (修改`config.py`中的`num_workers`)
   - 减小数据集大小进行初步实验

4. **聚类质量不佳**：
   - 增加训练轮数以改进特征提取
   - 尝试使用`--optimize_clusters`自动寻找最佳聚类数量
   - 考虑手动调整`eps`和`min_samples`参数 (修改`config.py`)

## 高级自定义

如需进一步自定义系统行为，可直接编辑`photo_classifier/config.py`文件中的配置项：

```python
CONFIG = {
    # 数据路径
    'data_dir': './data/photos',
    'output_dir': './output',
    'models_dir': './models',
    
    # 预处理参数
    'image_size': 224,  # 可调整以节省内存
    'batch_size': 64,   # 可调整以适应GPU内存
    'num_workers': 4,   # 可增加以加速数据加载
    
    # 特征提取和对比学习参数
    'feature_dim': 512,
    'projection_dim': 128,
    'temperature': 0.1,
    'learning_rate': 0.0003,
    'weight_decay': 1e-5,
    'epochs': 20,
    'pretrained_model': 'resnet50',  # 可改为'resnet18'以节省内存
    
    # 聚类参数
    'n_clusters': 20,  # 初始估计的类别数，会自动优化
    'eps': 0.5,        # DBSCAN的邻域大小参数
    'min_samples': 5,  # DBSCAN的最小样本数
    
    # 对焦检测参数 
    'laplacian_threshold': 100,  # 拉普拉斯算子方差阈值
    'fft_threshold': 10,         # 频域能量比阈值
    
    # 设备配置
    'device': 'cuda',
    'mixed_precision': True,     # 使用混合精度训练
}
```

## 示例

### 训练示例

```bash
# 基本训练，使用默认参数
./run.sh --mode train --data_dir ./data/photos

# 高级训练，自定义参数
./run.sh --mode train --data_dir ./data/photos --output_dir ./output/family_photos --optimize_clusters --batch_size 32 --epochs 30 --visualization
```

### 预测示例

```bash
# 对新照片进行分类
./run.sh --mode predict --data_dir ./data/new_photos

# 自定义对焦阈值
./run.sh --mode predict --data_dir ./data/new_photos --focus_threshold 150
```

### 增量示例

```bash
# 处理新照片，不微调模型
./run.sh --mode incremental --data_dir ./data/new_photos --epochs 0

# 处理新照片并微调模型
./run.sh --mode incremental --data_dir ./data/new_photos --epochs 5 --visualization
```

## 开发者信息

本系统基于以下开源项目和研究：

- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)