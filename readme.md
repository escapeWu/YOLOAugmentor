# 目标检测数据增强工具

基于：https://blog.paperspace.com/data-augmentation-for-bounding-boxes/ 进行二次封装

基于YOLO格式的自动化数据增强解决方案，支持多种增强策略和数据集划分。

## 🚀 主要功能

- ✅ **多格式支持**：同时兼容YOLO txt格式和LabelMe json格式标注
- 🎯 **增强策略**：水平翻转、随机缩放、旋转平移等数据增强方法
- 📊 **智能划分**：一键划分训练集/验证集/测试集（支持自定义比例）
- 📈 **可视化调试**：实时预览增强效果，支持键盘交互操作


## 🛠️ 快速开始

```python
# 初始化增强器（请修改为实际路径）
augmentor = YOLOAugmentor(
    img_dir=r"C:\Your\Image\Directory",
    label_dir=r"C:\Your\Label\Directory",
    output_dir=r"C:\Output\Directory",
    class_mapping={'object_class1': 0, 'object_class2': 1}  # 类别名称到ID的映射
)

# 配置增强流水线
aug_sequence = [
    augmentor.horizontal_flip(0.7),          # 70%概率水平翻转
    augmentor.scale(-0.1, 0.1),              # 缩放范围[-10%, +10%]
    augmentor.random_rotate(-5, 5),          # 随机旋转角度范围
    augmentor.random_translate((0, 0.3), (0, 0.3))  # 平移范围设置
]

# 生成增强数据（每个原始图片生成10个增强样本）
augmentor.process(aug_sequence, num_augments=10)

# 预览增强结果
augmentor.showFirstOutput()

# 划分数据集（7:2:1比例）
augmentor.collect(train=0.7, val=0.2, test=0.1)
```

## 📂 目录结构

```
output/
├── train/
│   ├── images/
│   └── labels/
├── val/
└── test/
```
