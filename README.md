# QSAR Prediction of BBB Permeability Based on Machine Learning upon a Novel Dataset of PET Tracers

<div align="center">

基于PET示踪剂数据集的机器学习血脑屏障渗透性QSAR预测模型

![Python](https://img.shields.io/badge/Python-3.8-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

## 📖 项目简介

本项目基于全新的PET示踪剂生物分布数据集(PETBD)，通过机器学习方法构建血脑屏障渗透性的QSAR预测模型。主要特点：

- 📊 **首个PET示踪剂数据集**: 包含816个小分子PET示踪剂的体内生物分布数据
- 🧬 **多器官分布数据**: 包含14个典型器官在静脉注射后60分钟的药物浓度数据
- 🧠 **BBB渗透性预测**: 预测化合物的血脑屏障渗透系数(logBB)和脑部药物浓度
- 🔄 **数据整合工具**: 用于整合公开文献中的PET示踪剂研究数据
- ✅ **体内验证**: 使用6个示踪剂的小鼠实验数据进行独立验证

## 🛠️ 环境配置

```bash
# 创建并激活环境
conda create -n MDM python=3.8
conda activate MDM

# 安装基础依赖
conda install numpy pandas scikit-learn rdkit

# 安装机器学习框架
pip install optuna xgboost lightgbm catboost

# 安装数据处理工具
pip install openpyxl==3.0.10
pip install "pandas<2.0.0"
pip install PyYAML==6.0
pip install rdkit_pypi==2022.9.5
pip install tqdm==4.64.1
pip install xlrd==2.0.1
pip install mordred
```

## 📁 项目结构

```
├── MachineLearningModels/    # 机器学习模型
│   ├── logBBModel/          # logBB预测模型
│   ├── CbrainModel/         # 脑部浓度预测模型
│   └── VivoValidation/      # 体内验证
├── MedicalDatasetsMerger/   # PET数据集整合工具
├── utils/                   # 工具类
├── preprocess/              # 数据预处理
├── data/                    # 数据存储
├── model/                   # 预训练模型存储
└── result/                  # 结果输出
```

## 📊 评估指标

模型评估使用以下指标：
| 指标 | 描述 |
|------|------|
| MSE | 均方误差 |
| RMSE | 均方根误差 |
| R² | 决定系数 |
| Adjusted R² | 调整R² |
| MAE | 平均绝对误差 |
| MAPE | 平均绝对百分比误差 |

## 💻 使用说明

### 数据准备
1. 将数据文件 (pet.xlsx) 放在 data/ 目录下
2. 数据文件必须包含以下列：
   - SMILES: 分子的SMILES表示
   - cbrain: 脑部浓度值
   - logBB: 血脑屏障渗透系数(可选)

### 模型训练
```bash
# 训练logBB预测模型
python MachineLearningModels/logBBModel/train.py

# 训练脑部浓度预测模型
python MachineLearningModels/CbrainModel/train.py
```

### 模型验证
```bash
# 体内验证
cd MachineLearningModels/VivoValidation
python Vivo_XGB_FP.py      # XGBoost模型(分子指纹)
python vivo_rf_Mordred.py  # 随机森林模型(Mordred描述符)
```

### 结果输出
- 预测结果保存在 result/vivoresult/ 目录下
- 日志文件保存在 data/log/ 目录下
- 模型文件保存在 model/ 目录下

## 🔍 特征说明

### 分子指纹
- Morgan指纹 (ECFP4)
- 1024位二进制指纹
- 半径为2的圆形指纹

### Mordred描述符
- 2D分子描述符
- 包括拓扑学特征、物理化学性质等
- 自动去除高度相关特征

## ⚙️ 模型参数

所有模型都经过Optuna自动调参优化，主要参数包括：
- 学习率
- 树的深度
- 特征采样比例
- 正则化参数
- 迭代次数

## 📝 日志记录

系统自动记录以下信息：
- 数据加载和预处理过程
- 特征计算和选择过程
- 模型训练和预测过程
- 评估指标和结果
- 错误和异常信息

## 🔧 错误处理

常见问题及解决方案：
1. 特征数量不匹配
   - 确保特征选择数量为50个
   - 检查预处理步骤是否正确

2. 模型文件缺失
   - 确保模型文件在正确路径下
   - 检查模型文件名是否正确

3. 内存不足
   - 减少批处理数据量
   - 使用数据生成器

## 📧 联系方式

如有任何问题，请联系:
- zhou_wei@gdut.edu.cn (W.Z.)
- zhiyundu@gdut.edu.cn (Z.D.)

## 📝 引用

如果您使用了本项目的代码或数据集，请引用我们的论文：

```bibtex
@article{su2024qsar,
  title={QSAR Prediction of BBB Permeability Based on Machine Learning upon a Novel Dataset of PET Tracers},
  author={Su, Qing and Ye, Zhilong and Han, Chunyan and Xiao, Ganyao and Chen, Jiazhi and Chen, Haiyan and Huang, Shun and Wang, Lu and Zhou, Wei and Du, Zhiyun},
  journal={},
  year={2024}
}
```