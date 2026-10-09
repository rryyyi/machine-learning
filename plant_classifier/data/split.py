"""训练集与验证集划分。"""

import random
from typing import Sequence


def stratified_split_indices(
    labels: Sequence[int], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """按类别分层切分，避免小数据集随机划分漏掉类别。"""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 0 和 1 之间")

    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label, indices in sorted(grouped.items()):
        if len(indices) < 2:
            raise ValueError(f"类别 {label} 至少需要两张图片才能分层划分")
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        val_count = min(max(1, round(len(shuffled) * val_ratio)), len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    return sorted(train_indices), sorted(val_indices)
