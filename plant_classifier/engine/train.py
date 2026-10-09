"""训练植物种类分类器，并保存验证集表现最好的模型。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from ..data import PlantImageDataset, load_plant_data, stratified_split_indices
from ..models.resnet18 import build_model, make_eval_transform, make_train_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练五类植物图像分类模型")
    parser.add_argument("--data", required=True, help="原始 ZIP 文件或解压后的数据目录")
    parser.add_argument(
        "--output", default="outputs/resnet18_best.pth", help="最佳模型检查点路径"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="验证损失早停轮数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 ImageNet 预训练权重；离线时可传 --no-pretrained",
    )
    parser.add_argument(
        "--fine-tune-all",
        action="store_true",
        help="解冻整个预训练网络（建议使用较小 learning-rate）",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model, loader, criterion, optimizer, device, freeze_early_batch_norm: bool
) -> tuple[float, float]:
    model.train()
    if freeze_early_batch_norm:
        # 冻结层的 BatchNorm 统计量也保持冻结，避免小批量训练破坏预训练特征。
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_count

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_count

    return total_loss / total_count, total_correct / total_count


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("epochs、batch-size 必须为正数，image-size 不得小于 32")
    if args.num_workers < 0 or args.patience < 1 or args.learning_rate <= 0:
        raise ValueError("num-workers 不能为负数，patience 与 learning-rate 必须为正数")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_plant_data(args.data)
    labels = [record.label for record in data.train_records]
    if any(label is None for label in labels):
        raise ValueError("训练数据中存在没有类别标签的图片")
    train_indices, val_indices = stratified_split_indices(
        [int(label) for label in labels], val_ratio=args.val_ratio, seed=args.seed
    )

    print(f"设备：{device}")
    print(f"类别（按文件夹名排序）：{', '.join(data.classes)}")
    print(f"训练图片：{len(train_indices)}；验证图片：{len(val_indices)}；测试图片：{len(data.test_records)}")
    if device.type == "cpu":
        print("提示：当前使用 CPU；若本机有 CUDA，可安装匹配版本的 PyTorch 以加快训练。")

    train_dataset = PlantImageDataset(
        data.train_records,
        transform=make_train_transform(args.image_size),
        zip_path=data.zip_path,
    )
    val_dataset = PlantImageDataset(
        data.train_records,
        transform=make_eval_transform(args.image_size),
        zip_path=data.zip_path,
    )
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        len(data.classes),
        pretrained=args.pretrained,
        fine_tune_all=args.fine_tune_all,
    ).to(device)
    if args.pretrained:
        backbone_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("fc.")
        ]
        head_parameters = list(model.fc.parameters())
        optimizer = AdamW(
            [
                {"params": backbone_parameters, "lr": args.learning_rate},
                {"params": head_parameters, "lr": args.learning_rate * 10},
            ],
            weight_decay=1e-4,
        )
    else:
        # 无预训练权重时必须训练整个网络，不能冻结随机初始化的特征层。
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.pretrained and not args.fine_tune_all,
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lrs = ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.3f} | lr {current_lrs}"
        )

        # 以验证损失选择检查点，避免只追逐小验证集上的单次准确率波动。
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_name": "resnet18",
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "class_names": list(data.classes),
                "image_size": args.image_size,
                "pretrained": args.pretrained,
                "fine_tune_all": args.fine_tune_all,
                "best_epoch": best_epoch,
                "val_loss": best_val_loss,
                "val_accuracy": best_val_accuracy,
                "seed": args.seed,
            }
            torch.save(checkpoint, output_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证损失连续 {args.patience} 轮未改善，提前停止。")
                break

    print(
        f"最佳模型：{output_path}（epoch={best_epoch}, "
        f"val_loss={best_val_loss:.4f}, val_acc={best_val_accuracy:.3f}）"
    )


if __name__ == "__main__":
    main()
