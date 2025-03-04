# Medical Datasets Merger
用于整合从不同文献中收集而来的药物浓度数据的Python脚本


## 使用方法

1. 克隆本仓库
2. 创建新python环境(以conda环境，命名为 MDM 为例)
    > conda create -n MDM python=3.7
3. 在根目录中添加环境依赖
    > conda activate MDM
    > 
    > conda install --yes --file requirements.txt
4. 在根目录中添加data目录，并将原始数据集（包含mol与浓度excel文件）
5. 运行run.py文件（目前还没做命令行运行，需要使用IDE运行）
6. 整合结果将保存在 result/"当前日期” 目录下