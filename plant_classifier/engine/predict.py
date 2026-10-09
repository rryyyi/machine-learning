"""使用训练好的检查点预测测试图片并生成课程提交 CSV。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import (
    PlantImageDataset,
    load_plant_data,
    order_test_records,
    read_template_ids,
)
from ..models.resnet18 import build_model, make_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成植物分类 submission.csv")
    parser.add_argument("--data", required=True, help="原始 ZIP 文件或解压后的数据目录")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="一个或多个 train.py 检查点；多个模型将平均预测概率",
    )
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--template", default=None, help="可选：显式指定 submission CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="平均原图及水平/垂直翻转视图的概率",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    # 检查点只包含张量和基础 Python 类型；新版本优先使用安全权重加载模式。
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # 兼容较旧的 PyTorch 版本，它们还没有 weights_only 参数。
        return torch.load(path, map_location=device)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size 必须为正数，num-workers 不能为负数")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_plant_data(args.data)
    checkpoints = []
    models = []
    for checkpoint_name in args.checkpoint:
        checkpoint_path = Path(checkpoint_name).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"模型检查点不存在：{checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        model_classes = tuple(checkpoint["class_names"])
        if model_classes != data.classes:
            raise ValueError(
                "模型类别顺序与当前训练目录不一致。"
                f"模型={model_classes}，数据={data.classes}"
            )
        checkpoints.append(checkpoint)
        model = build_model(
            len(model_classes),
            pretrained=False,
            model_name=checkpoint.get("model_name", "resnet18"),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
    class_names = tuple(checkpoints[0]["class_names"])
    image_sizes = {int(checkpoint.get("image_size", 224)) for checkpoint in checkpoints}
    if len(image_sizes) != 1:
        raise ValueError("集成检查点必须使用相同的 image_size")

    template_ids = read_template_ids(data, args.template)
    ordered_records, output_ids = order_test_records(data.test_records, template_ids)
    dataset = PlantImageDataset(
        ordered_records,
        transform=make_eval_transform(image_sizes.pop()),
        zip_path=data.zip_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    predicted_indices: list[int] = []
    for images, _names in loader:
        images = images.to(device, non_blocking=True)
        views = (images, torch.flip(images, [3]), torch.flip(images, [2]))
        if args.tta:
            views = (*views, torch.flip(images, [2, 3]))
        ensemble_probabilities = []
        with torch.no_grad():
            for model in models:
                view_probabilities = [model(view).softmax(dim=1) for view in views]
                ensemble_probabilities.append(torch.stack(view_probabilities).mean(0))
        probabilities = torch.stack(ensemble_probabilities).mean(0)
        predicted_indices.extend(probabilities.argmax(dim=1).cpu().tolist())

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Category"])
        for identifier, class_index in zip(output_ids, predicted_indices, strict=True):
            writer.writerow([identifier, class_names[class_index]])

    print(f"已写入 {len(output_ids)} 条预测：{output_path}")
    print(
        f"使用 {len(models)} 个模型，TTA={'开' if args.tta else '关'}；"
        "提交表仅使用模板 ID。"
    )


if __name__ == "__main__":
    main()
