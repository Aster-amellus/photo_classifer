无监督照片分类与对焦检测系统使用指南
本系统使用无监督学习方法对大量日常照片进行自动分类，并能检测照片的对焦质量，专为本地RTX 4060环境优化。

1. 系统介绍
本系统主要功能：

- 自动对大量日常照片进行无监督分类
- 检测照片的对焦质量，区分清晰和模糊照片
- 自动优化分类的簇数
- 处理新增照片的增量学习功能
- 丰富的可视化工具展示分类结果
技术亮点：

- 使用预训练ResNet50作为特征提取器
- 采用SimCLR对比学习框架学习照片的表示
- 结合K-means和DBSCAN的混合聚类策略
- 双重验证的对焦检测算法
- 针对RTX 4060的性能优化

2. 安装指南
2.1 环境要求
Python 3.8+
NVIDIA RTX 4060 GPU
CUDA 11.7+ 和 cuDNN
8GB+ RAM

2.2 安装步骤
克隆项目代码
```bash
git clone https://github.com/yourusername/photo-classifier.git
cd photo-classifier
```

创建虚拟环境
```bash
python -m venv .venv
```

激活虚拟环境
```bash
# 在Windows上
.venv\Scripts\activate

# 在Linux/Mac上
source .venv/bin/activate
```

安装依赖
```bash
pip install -r requirements.txt
```

创建项目目录结构
```bash
mkdir -p data/photos
mkdir -p output
mkdir -p models
```

安装CUDA和cuDNN 根据您的系统，请按照NVIDIA官方文档安装CUDA和cuDNN。确保安装的CUDA版本与PyTorch兼容。

3. 项目结构
```
photo_classifier/
├── config.py              # 配置文件
├── main.py                # 主程序
├── models/                # 模型定义
│   └── simclr.py          # SimCLR模型
├── utils/                 # 工具函数
│   ├── clustering/        # 聚类相关
│   ├── focus/             # 对焦检测
│   └── preprocessing/     # 图像预处理
data/                      # 数据目录
├── photos/                # 存放照片的目录
models/                    # 保存的模型
output/                    # 输出结果
├── logs/                  # 训练日志
├── organized_photos/      # 分类后的照片
└── visualizations/        # 可视化结果
```

4. 使用方法
4.1 放置照片
将您要分类的照片放入data/photos目录中。系统将自动递归搜索该目录下的所有图片文件。

4.2 训练模式
训练模式用于首次处理照片集，会训练模型并进行聚类。

基本用法：
```bash
# 直接在项目目录下运行
python main.py --mode train
```

高级选项：
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

参数说明：
```
   --data_dir: 照片所在目录，默认为config.py中设置的路径
   --output_dir: 输出目录，默认为config.py中设置的路径
   --n_clusters: 初始聚类数量，如果使用--optimize_clusters则为最大聚类数量
   --batch_size: 训练批大小，根据GPU内存调整
   --epochs: 训练轮数
   --optimize_clusters: 是否自动优化聚类数量
   --focus_threshold: 自定义对焦阈值
```

4.3 预测模式
对新照片进行分类，不再重新训练模型。

```bash
# 直接在项目目录下运行
python main.py \
  --mode predict \
  --data_dir path/to/new_photos \
  --model_path final_model.pth
```

4.4 增量学习模式
处理新添加的照片，并可选择性地更新模型。

```bash
# 直接在项目目录下运行
python main.py \
  --mode incremental \
  --data_dir path/to/photos \
  --model_path final_model.pth
```

5. 输出结果
5.1 分类结果
分类后的照片将存放在`output/organized_photos/`目录下，按照聚类标签组织。对焦质量较低的照片会被放入每个聚类的low_focus子目录。

5.2 可视化结果
`output/visualizations/feature_visualization_tsne.png`: t-SNE降维的特征可视化
`output/visualizations/feature_visualization_pca.png`: PCA降维的特征可视化
`output/visualizations/interactive_visualization_tsne.html`: 交互式可视化(需安装plotly)
`output/clustering/cluster_metrics.png`: 聚类指标评估图表

5.3 模型文件
训练好的模型将保存在models/目录下：

`best_model.pth`: 训练过程中表现最好的模型
`final_model.pth`: 训练结束时的模型
`cluster_model.pkl`: 聚类模型
`updated_model_*.pth`: 增量学习更新后的模型

6. 性能优化
为了在RTX 4060上获得最佳性能：

批大小调整：默认批大小为64，如遇内存不足，可尝试减小批大小
混合精度训练：系统默认使用混合精度训练，加速训练过程
预处理优化：图像预处理使用GPU加速
并行数据加载：利用多线程加载数据，提高GPU利用率

7. 故障排除
问题: 内存溢出错误 解决方案: 减小批大小，在`config.py`中将batch_size从64减至32或16

问题: 聚类结果不理想 解决方案: 尝试使用--optimize_clusters参数自动优化聚类数量

问题: 对焦检测不准确 解决方案: 通过--focus_threshold参数调整对焦阈值，较大的值会使系统更严格

问题: CUDA相关错误 解决方案: 确保CUDA版本与PyTorch兼容，检查GPU驱动是否正确安装

8. 自定义配置
您可以通过修改`config.py`文件来自定义系统的配置参数。主要配置项包括：

图像大小和预处理参数
模型架构和训练参数
聚类参数
对焦检测阈值
设备配置（GPU/CPU）

9. 适用场景
该系统特别适用于以下场景：

整理大量个人照片集（家庭相册、旅游照片等）
筛选出清晰度高的照片
自动发现照片的自然分组
持续处理新增照片

10. 注意事项
系统首次处理大量照片时会占用较多计算资源，请确保电脑散热良好
建议首次使用时先用小批量照片（100-200张）测试系统的行为
增量模式会记录已处理的照片，请勿手动删除output目录下的processed_images.txt文件

希望这个系统能帮助您轻松整理照片集！如有任何问题，请随时联系。