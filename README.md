# 东北大学机器学习课设一：植物种类分类

本项目使用深度学习模型，根据植物图片预测其**种类**。数据集中的图片可能覆盖不同生长阶段，但当前标签只有植物种类，没有生长阶段标签；因此模型不能监督学习“生长状态”分类。

## 数据集核对

压缩包 `2026task1.zip` 中的文件夹和 CSV 文件名带有 `task2`，本项目按你的说明将其视为 task1 数据。脚本根据 `train/`、`test/` 目录结构自动定位数据，无需重命名或解压。检查结果为 5 个类别、每类 100 张训练图，以及 378 张测试图。类别从训练目录名读取：

- Black-grass
- Common wheat
- Loose Silky-bent
- Scentless Mayweed
- Sugar beet

压缩包附带的 `submission-for-task2.csv` 表头为 `ID,Category`，且 `Category` 列已有内容。为避免把测试标签泄漏给模型，程序**只读取 ID 列**来校验、排序图片；原有 `Category` 值不会用于训练或预测，生成提交文件时会由模型预测值覆盖。

## 算法思路

1. 使用 ImageNet 预训练的 ResNet18 作为卷积神经网络特征提取器，将最后的全连接层替换为 5 类分类头。
2. 训练数据按类别分层切分为 80% 训练集和 20% 验证集，固定随机种子，防止小样本随机划分漏掉类别。
3. 通过随机裁剪、水平/垂直翻转、轻微旋转和颜色扰动扩充训练变化；验证和测试阶段采用固定缩放与中心裁剪。
4. 使用交叉熵损失和 AdamW 优化器。迁移学习时冻结较低层，只微调 ResNet18 的高层与分类头；根据验证损失调整学习率，并在验证损失长期不改善时提前停止。
5. 保存验证损失最低的检查点；预测阶段严格按提交模板 `ID` 顺序输出 `ID,Category` 两列。

这些增强和验证仅用于缓解每类 100 张图片带来的过拟合风险。最终效果仍应以实际运行的验证集指标和课程评测为准。

## 运行方法

建议使用 Python 3.10 或更新版本，并安装与你的 CUDA/CPU 环境相匹配的 PyTorch 与 torchvision：

```powershell
python -m pip install -r requirements.txt
```

训练（直接读取原始 ZIP，不会将数据加入 Git）：

```powershell
python train.py --data "C:\Users\Yukinoshita\Downloads\2026task1.zip"
```

首次使用预训练模型时，torchvision 可能需要下载 ImageNet 权重；训练结束后模型默认保存到 `outputs/resnet18_best.pth`。如果环境不能联网，可用 `--no-pretrained` 运行，但小样本从头训练通常效果较弱：

```powershell
python train.py --data "C:\Users\Yukinoshita\Downloads\2026task1.zip" --no-pretrained
```

生成提交文件：

```powershell
python predict.py --data "C:\Users\Yukinoshita\Downloads\2026task1.zip" --checkpoint outputs/resnet18_best.pth --output outputs/submission.csv
```

输出默认保存在 `outputs/submission.csv`，格式为：

```csv
ID,Category
图片文件名.png,模型预测的植物类别
```

也可以将 `--data` 指向解压后的文件夹；脚本会递归寻找同一层级下的 `train/` 和 `test/`。如需指定其他模板，可添加 `--template 路径`。有 CUDA 时自动使用 GPU；否则使用 CPU。

## 检查

运行单元测试：

```powershell
python -m unittest discover -s tests -v
```
