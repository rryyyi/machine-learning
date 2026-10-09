# Implementation Notes — 植物种类分类课设

## Plan baseline

- **Plan / spec source**：本次用户请求、随附课设描述，以及仓库中的 `agent.md`。
- **Key assumptions**：实现的是带类别目录的监督式图像分类；测试集提交格式是 `ID,Category`；数据集只用于本地运行，不提交至 Git；深度学习框架使用 PyTorch。

## Deviations

| # | Location / trigger | Original plan | What was done (conservative option) | Why | Impact / to confirm |
|---|--------------------|---------------|--------------------------------------|-----|---------------------|
| 1 | `2026task1.zip` 数据布局 | 按 task1 命名定位数据 | 由 `train/`、`test/` 结构定位 ZIP 内数据，不改动或重命名原始压缩包 | 压缩包内部目录与 CSV 名带 `task2`，用户说明其实际是 task1 | 无需手工解压或改名 |
| 2 | 提交模板 `Category` 列 | 模板可能为空 | 仅读取模板 `ID` 列，完全忽略其 `Category` 值 | 模板中的 `Category` 已有内容，不能作为测试监督信号 | 防止测试标签泄漏 |
| 3 | 本机训练环境 | 优先 GPU 训练 | 自动检测 CUDA，并保留 CPU 路径 | 当前环境只有 CPU 版 PyTorch | CPU 训练会较慢；可在有匹配 CUDA 环境的机器训练 |
| 4 | 项目层次 | 入口脚本和算法模块位于仓库根目录 | 按 `plant_classifier/data/`、`models/`、`engine/` 拆分职责，并把实验说明和实现记录归入 `docs/` | 用户要求采用层次清楚的深度学习项目结构 | 运行方式改为 `python -m plant_classifier.engine.train` / `predict` |

## New unknowns that surfaced

- 类别目录是生物类别标签而非生长状态标签；若课设还要求生长状态分类，需要额外提供逐图生长阶段标签。
- 预训练权重首次运行可能需要联网下载；离线流程提供 `--no-pretrained`，但可能牺牲准确率。

## Decisions that need a human

- [ ] 如果老师实际需要“生长状态”作为预测目标，而不仅是植物种类，需要补充阶段标签及其定义。
