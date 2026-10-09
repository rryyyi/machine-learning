# 东北大学机器学习课设一：植物种类分类

使用 PyTorch 和 ResNet18 迁移学习，从植物图片预测植物**种类**。目前数据没有生长阶段标签，因此本项目不训练生长状态分类器。

## 项目结构

```text
机器学习 2/
├── plant_classifier/       # 可复用的分类算法包
│   ├── data/
│   │   ├── dataset.py      # ZIP/目录读取与 Dataset
│   │   ├── split.py        # 分层训练/验证集划分
│   │   └── submission.py   # 模板 ID 对齐和校验
│   ├── models/
│   │   └── resnet18.py     # ResNet18 与图像预处理
│   └── engine/
│       ├── train.py        # 训练、验证、早停、保存检查点
│       └── predict.py      # 加载检查点并生成 submission.csv
├── tests/                  # 自动化测试
├── docs/                   # 实验方案和实现记录
├── .gitignore
├── agent.md                # 仓库协作约定
├── pyproject.toml          # Python 包与命令入口配置
├── requirements.txt
└── README.md               # 项目入口说明
```

各层职责分开：`plant_classifier/` 放算法实现，`tests/` 验证数据与提交逻辑，`docs/` 保存实验解释；项目根目录只保留安装、运行和导航文件。

## 数据检查

压缩包 `2026task1.zip` 内部目录和 CSV 名称带有 `task2`，按你的说明视为 task1 数据。程序按 `train/`、`test/` 结构自动定位，可直接读 ZIP，不必重命名或解压。核对结果为 5 类、每类 100 张训练图、378 张测试图。

压缩包中的 `submission-for-task2.csv` 格式为 `ID,Category`，且 `Category` 列已有内容。为避免测试标签泄漏，程序仅读取 `ID` 来匹配图片和保持提交顺序，忽略模板里原有的 `Category` 值。

## 安装和运行

建议使用 Python 3.10 或更新版本，并按本机 CPU/CUDA 环境安装匹配的 PyTorch 与 torchvision：

```powershell
python -m pip install -r requirements.txt
```

从项目根目录启动训练。默认冻结较低层；为复现当前候选提交使用的全网络低学习率微调配置：

```powershell
python -m plant_classifier.engine.train --data "C:\Users\Yukinoshita\Downloads\2026task1.zip" --output outputs/resnet18_all_best.pth --epochs 15 --batch-size 32 --patience 5 --learning-rate 2e-5 --fine-tune-all
```

首次使用预训练权重时，torchvision 可能需要联网下载；离线时可以关闭预训练（小样本从头训练的效果通常较弱）：

```powershell
python -m plant_classifier.engine.train --data "C:\Users\Yukinoshita\Downloads\2026task1.zip" --no-pretrained
```

生成提交文件：

```powershell
python -m plant_classifier.engine.predict --data "C:\Users\Yukinoshita\Downloads\2026task1.zip" --checkpoint outputs/resnet18_all_best.pth --output outputs/submission.csv
```

输出列为 `ID,Category`，顺序与 ZIP 中模板的 `ID` 一致。也可以将 `--data` 指向解压目录；若要使用其他提交模板，添加 `--template 路径`。有 CUDA 时自动使用 GPU，否则使用 CPU。

如需安装包并使用 `plant-train` / `plant-predict` 命令，可在安装依赖后执行：

```powershell
python -m pip install -e . --no-deps
plant-train --data "C:\Users\Yukinoshita\Downloads\2026task1.zip" --output outputs/resnet18_all_best.pth --epochs 15 --batch-size 32 --patience 5 --learning-rate 2e-5 --fine-tune-all
plant-predict --data "C:\Users\Yukinoshita\Downloads\2026task1.zip" --checkpoint outputs/resnet18_all_best.pth
```

## 验证

```powershell
python -m unittest discover -s tests -v
```

实验方法和适用范围见 [docs/experiment.md](docs/experiment.md)。
