# 无监督照片分类与对焦检测系统使用指南

本系统使用无监督学习方法对大量日常照片进行自动分类，并能检测照片的对焦质量，专为本地RTX 4060环境优化。

## 1. 系统介绍

### 主要功能：

- 自动对大量日常照片进行无监督分类
- 检测照片的对焦质量，区分清晰和模糊照片
- 自动优化分类的簇数
- 处理新增照片的增量学习功能
- 丰富的可视化工具展示分类结果
- GPU加速的对焦质量分析，提高处理效率

### 技术亮点：

- 使用预训练EfficientNet-B3作为特征提取器
- 采用MoCo v2对比学习框架学习照片的表示
- 结合K-means和DBSCAN的混合聚类策略
- 多重算法融合的对焦检测系统，支持GPU加速
- 针对RTX 4060的性能优化
- 批量处理技术大幅提升处理速度

## 2. 安装指南

### 2.1 环境要求

- Python 3.8+
- NVIDIA RTX 4060 GPU
- CUDA 11.7+ 和 cuDNN
- 8GB+ RAM

### 2.2 安装步骤

#### 克隆项目代码
```bash
git clone https://github.com/yourusername/photo-classifier.git
cd photo-classifier
```

#### 创建虚拟环境
```bash
python -m venv .venv
```

#### 激活虚拟环境
```bash
# 在Windows上
.venv\Scripts\activate

# 在Linux/Mac上
source .venv/bin/activate
```

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 创建项目目录结构
```bash
mkdir -p data/photos
mkdir -p output
mkdir -p models
```

#### 安装CUDA和cuDNN
根据您的系统，请按照NVIDIA官方文档安装CUDA和cuDNN。确保安装的CUDA版本与PyTorch兼容。

## 3. 项目结构
```
photo_classifier/
├── config.py              # 配置文件
├── main.py                # 主程序
├── models/                # 模型定义
│   └── simclr.py          # SimCLR模型
├── utils/                 # 工具函数
│   ├── clustering/        # 聚类相关
│   ├── focus/             # 对焦检测
│   │   ├── focus_detection.py  # 对焦检测核心
│   │   ├── focus_utils.py      # 对焦工具函数
│   │   ├── focus_demo.py       # 对焦演示程序
│   │   └── focus_calibration.py # 对焦校准工具
│   ├── preprocessing/     # 图像预处理
│   └── trainer.py         # 模型训练器
├── data/                  # 数据目录
│   └── photos/            # 存放照片的目录
├── models/                # 保存的模型
└── output/                # 输出结果
    ├── logs/              # 训练日志
    ├── organized_photos/  # 分类后的照片
    └── visualizations/    # 可视化结果
```

## 4. 使用方法

### 4.1 放置照片
将您要分类的照片放入`data/photos`目录中。系统将自动递归搜索该目录下的所有图片文件。

### 4.2 训练模式
训练模式用于首次处理照片集，会训练模型并进行聚类。

#### 基本用法：
```bash
# 直接在项目目录下运行
python main.py --mode train
```

#### 高级选项：
```bash
# 直接在项目目录下运行
python main.py \
  --mode train \
  --data_dir path/to/photos \
  --output_dir path/to/output \
  --n_clusters 20 \
  --batch_size 32 \
  --epochs 10 \
  --optimize_clusters
```

#### 参数说明：
```
   --data_dir: 照片所在目录，默认为config.py中设置的路径
   --output_dir: 输出目录，默认为config.py中设置的路径
   --n_clusters: 初始聚类数量，如果使用--optimize_clusters则为最大聚类数量
   --batch_size: 训练批大小，根据GPU内存调整
   --epochs: 训练轮数
   --optimize_clusters: 是否自动优化聚类数量
   --focus_threshold: 自定义对焦阈值
```

### 4.3 预测模式
对新照片进行分类，不再重新训练模型。

```bash
# 直接在项目目录下运行
python main.py \
  --mode predict \
  --data_dir path/to/new_photos \
  --model_path final_model.pth
```

### 4.4 增量学习模式
处理新添加的照片，并可选择性地更新模型。

```bash
# 直接在项目目录下运行
python main.py \
  --mode incremental \
  --data_dir path/to/photos \
  --model_path final_model.pth
```

### 4.5 对焦检测工具

#### 对焦演示程序
用于测试和展示对焦检测功能：

```bash
# 分析单张图像
python utils/focus/focus_demo.py \
  --mode single \
  --input path/to/image.jpg \
  --output path/to/output_dir

# 批量分析图像
python utils/focus/focus_demo.py \
  --mode batch \
  --input path/to/image_dir \
  --output path/to/output_dir

# 比较不同对焦检测方法
python utils/focus/focus_demo.py \
  --mode compare \
  --input path/to/image_dir \
  --output path/to/output_dir
```

#### 对焦阈值校准工具
用于优化对焦检测阈值，需要提供已知对焦良好和对焦不佳的图像：

```bash
python utils/focus/focus_calibration.py \
  --focused_dir path/to/focused_images \
  --unfocused_dir path/to/unfocused_images \
  --output path/to/calibration_output \
  --adaptive  # 启用自适应阈值
```

## 5. 输出结果

### 5.1 分类结果
分类后的照片将存放在`output/organized_photos/`目录下，按照聚类标签组织。对焦质量较低的照片会被放入每个聚类的`low_focus`子目录。

### 5.2 可视化结果
- `output/visualizations/feature_visualization_tsne.png`: t-SNE降维的特征可视化
- `output/visualizations/feature_visualization_pca.png`: PCA降维的特征可视化
- `output/visualizations/interactive_visualization_tsne.html`: 交互式可视化(需安装plotly)
- `output/clustering/cluster_metrics.png`: 聚类指标评估图表

### 5.3 模型文件
训练好的模型将保存在`models/`目录下：

- `best_model.pth`: 训练过程中表现最好的模型
- `final_model.pth`: 训练结束时的模型
- `cluster_model.pkl`: 聚类模型
- `updated_model_*.pth`: 增量学习更新后的模型

## 6. 性能优化

### GPU加速
- **对焦检测GPU加速**：使用CUDA加速对焦质量计算，速度提升5-10倍
- **批量处理**：一次处理多张图片，减少GPU与CPU之间的数据传输开销
- **混合精度训练**：系统默认使用混合精度训练，加速训练过程
- **高效模型结构**：优化的EfficientNet-B3模型提供更好的特征表示

### 内存优化
- **轻量级模式**：对于大量图片，可启用轻量级对焦检测模式，减少内存占用
- **流式处理**：大数据集采用流式处理，避免内存溢出
- **批量大小自适应**：根据GPU内存大小自动调整批量大小

### 速度优化
- **多线程数据加载**：利用多线程加载数据，提高GPU利用率
- **并行处理**：使用线程池并行处理多个任务
- **预加载数据**：训练期间预加载下一批数据，减少等待时间

## 7. 常见问题与解决方案

### 问题: 内存溢出错误
**解决方案**: 减小批大小，在`config.py`中将`batch_size`从64减至32或16，并启用轻量级对焦检测模式`light_focus_mode=True`。

### 问题: 聚类结果不理想
**解决方案**: 尝试使用`--optimize_clusters`参数自动优化聚类数量，或在`config.py`中增加`n_clusters`值。

### 问题: 对焦检测不准确
**解决方案**: 使用对焦校准工具`focus_calibration.py`根据您的照片特点调整对焦阈值，或通过`--focus_threshold`参数调整阈值。

### 问题: CUDA相关错误
**解决方案**: 确保CUDA版本与PyTorch兼容，检查GPU驱动是否正确安装。尝试执行`torch.cuda.is_available()`确认CUDA可用。

### 问题: 处理大量照片时系统卡顿
**解决方案**: 启用轻量级对焦检测模式，减少每批处理的图像数量，关闭其他占用GPU的程序。

## 8. 进阶使用

### 8.1 自定义配置
您可以通过修改`config.py`文件来自定义系统的配置参数：
- 调整`image_size`更改处理图像的大小
- 修改`pretrained_model`选择不同的特征提取器
- 调整`laplacian_threshold`和`fft_threshold`优化对焦检测
- 设置`use_gpu_focus=False`在CPU上运行对焦检测
- 调整`light_focus_mode`在速度和精度之间权衡

### 8.2 对焦检测的高级用法
对焦检测模块支持多种分析方法和可视化：
- 使用`focus_demo.py`的比较模式查看不同算法的结果
- 使用`analyze_focus_quality`函数获取详细的对焦质量报告
- 使用`batch_analyze_images`批量处理整个文件夹的图像
- 使用`calibrate_focus_thresholds`自动优化检测阈值

### 8.3 集成到其他系统
本系统的模块化设计使其易于集成到其他图像处理流程中：
- 对焦检测模块可独立使用
- 特征提取器可用于其他图像分析任务
- 聚类模块可用于其他数据集的聚类

## 9. 致谢与参考
- SimCLR和MoCo对比学习框架
- EfficientNet模型架构
- PyTorch深度学习框架
- OpenCV计算机视觉库

希望这个系统能帮助您轻松整理照片集！如有任何问题，请随时联系。