"""使用训练好的检查点预测测试图片并生成课程提交 CSV。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plant_data import (
    PlantImageDataset,
    load_plant_data,
    order_test_records,
    read_template_ids,
)
from plant_model import build_model, make_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成植物分类 submission.csv")
    parser.add_argument("--data", required=True, help="原始 ZIP 文件或解压后的数据目录")
    parser.add_argument("--checkpoint", required=True, help="train.py 保存的 .pth 文件")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--template", default=None, help="可选：显式指定 submission CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
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
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型检查点不存在：{checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = tuple(checkpoint["class_names"])
    if class_names != data.classes:
        raise ValueError(
            "模型类别顺序与当前训练目录不一致。"
            f"模型={class_names}，数据={data.classes}"
        )

    template_ids = read_template_ids(data, args.template)
    ordered_records, output_ids = order_test_records(data.test_records, template_ids)
    dataset = PlantImageDataset(
        ordered_records,
        transform=make_eval_transform(int(checkpoint.get("image_size", 224))),
        zip_path=data.zip_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predicted_indices: list[int] = []
    for images, _names in loader:
        logits = model(images.to(device, non_blocking=True))
        predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Category"])
        for identifier, class_index in zip(output_ids, predicted_indices, strict=True):
            writer.writerow([identifier, class_names[class_index]])

    print(f"已写入 {len(output_ids)} 条预测：{output_path}")
    print("提交表仅使用了模板 ID；模板中原有的 Category 值没有用于预测。")


if __name__ == "__main__":
    main()
