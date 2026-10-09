"""植物分类实验的数据发现、读取和提交文件处理。"""

from __future__ import annotations

import csv
import io
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    """单张图片及其标签；测试图片的 label 为 None。"""

    path: str | Path
    name: str
    label: int | None


@dataclass(frozen=True)
class PlantData:
    """从压缩包或目录中发现的训练集、测试集和可选提交模板。"""

    classes: tuple[str, ...]
    train_records: tuple[ImageRecord, ...]
    test_records: tuple[ImageRecord, ...]
    zip_path: Path | None = None
    template_path: Path | None = None
    template_member: str | None = None


class PlantImageDataset(Dataset):
    """统一读取磁盘目录和 ZIP 内图片，避免必须复制/解压大数据集。"""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform=None,
        zip_path: Path | None = None,
    ) -> None:
        self.records = tuple(records)
        self.transform = transform
        self.zip_path = zip_path
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        # Windows 多进程 DataLoader 会 pickle Dataset；每个 worker 自己打开 ZIP。
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def _get_archive(self) -> zipfile.ZipFile:
        if self.zip_path is None:
            raise RuntimeError("当前数据集不是从 ZIP 文件读取的")
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.zip_path, "r")
        return self._archive

    def _read_image(self, record: ImageRecord) -> Image.Image:
        if self.zip_path is not None:
            with self._get_archive().open(str(record.path), "r") as stream:
                with Image.open(stream) as image:
                    return image.convert("RGB")

        with Path(record.path).open("rb") as stream:
            with Image.open(stream) as image:
                return image.convert("RGB")

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self._read_image(record)
        if self.transform is not None:
            image = self.transform(image)

        if record.label is None:
            return image, record.name
        return image, record.label


def _is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _template_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*.csv")
            if path.name.casefold().startswith(("submission", "sample_submission"))
        ),
        key=lambda path: str(path).casefold(),
    )


def _load_directory_data(root: Path) -> PlantData:
    candidates = [root]
    candidates.extend(path for path in root.rglob("train") if path.is_dir())
    pairs = {
        (train_dir.resolve(), (train_dir.parent / "test").resolve())
        for train_dir in candidates
        if train_dir.name.casefold() == "train"
        and train_dir.is_dir()
        and (train_dir.parent / "test").is_dir()
    }
    if len(pairs) != 1:
        raise ValueError(
            "无法唯一定位 train/ 和 test/ 目录。请将 --data 指向包含这两个目录的"
            "数据集文件夹，或直接传入原始 ZIP 文件。"
        )

    train_dir, test_dir = next(iter(pairs))
    class_dirs = sorted(
        (path for path in train_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if len(class_dirs) < 2:
        raise ValueError(f"训练目录中至少需要两个类别文件夹：{train_dir}")

    classes = tuple(path.name for path in class_dirs)
    train_records: list[ImageRecord] = []
    for label, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            (path for path in class_dir.rglob("*") if path.is_file() and _is_image(path)),
            key=lambda path: str(path).casefold(),
        )
        train_records.extend(
            ImageRecord(path=path, name=path.name, label=label) for path in image_paths
        )

    test_paths = sorted(
        (path for path in test_dir.rglob("*") if path.is_file() and _is_image(path)),
        key=lambda path: str(path).casefold(),
    )
    if not train_records or not test_paths:
        raise ValueError(f"训练图片或测试图片为空：{train_dir}, {test_dir}")

    templates = _template_files(root)
    template_path = templates[0] if len(templates) == 1 else None
    if len(templates) > 1:
        raise ValueError(f"发现多个 submission CSV，请用 --template 明确指定：{templates}")

    return PlantData(
        classes=classes,
        train_records=tuple(train_records),
        test_records=tuple(
            ImageRecord(path=path, name=path.name, label=None) for path in test_paths
        ),
        template_path=template_path,
    )


def _load_zip_data(zip_path: Path) -> PlantData:
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]

    train_by_root: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    test_by_root: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        parts = PurePosixPath(name).parts
        if not parts or not _is_image(name):
            continue

        for index, part in enumerate(parts):
            if part.casefold() == "train" and index + 2 == len(parts) - 1:
                root = parts[:index]
                class_name = parts[index + 1]
                train_by_root.setdefault(root, []).append((class_name, name))
            elif part.casefold() == "test" and index == len(parts) - 2:
                root = parts[:index]
                test_by_root.setdefault(root, []).append(name)

    roots = [root for root in train_by_root if root in test_by_root]
    if len(roots) != 1:
        raise ValueError(
            "无法唯一定位 ZIP 中同时包含 train/ 和 test/ 的数据集目录；"
            "请检查压缩包结构。"
        )

    root = roots[0]
    class_names = sorted(
        {class_name for class_name, _ in train_by_root[root]}, key=str.casefold
    )
    if len(class_names) < 2:
        raise ValueError("ZIP 训练集至少需要两个类别")
    class_to_label = {name: label for label, name in enumerate(class_names)}

    train_items = sorted(
        train_by_root[root], key=lambda item: (item[0].casefold(), item[1].casefold())
    )
    test_names = sorted(test_by_root[root], key=str.casefold)
    if not train_items or not test_names:
        raise ValueError("ZIP 中训练图片或测试图片为空")

    template_candidates = [
        name
        for name in names
        if PurePosixPath(name).suffix.casefold() == ".csv"
        and PurePosixPath(name).name.casefold().startswith(
            ("submission", "sample_submission")
        )
    ]
    template_member = None
    if template_candidates:
        shallowest = min(len(PurePosixPath(name).parts) for name in template_candidates)
        closest = [
            name
            for name in template_candidates
            if len(PurePosixPath(name).parts) == shallowest
        ]
        if len(closest) != 1:
            raise ValueError(f"ZIP 中发现多个提交模板：{closest}")
        template_member = closest[0]

    return PlantData(
        classes=tuple(class_names),
        train_records=tuple(
            ImageRecord(
                path=name,
                name=PurePosixPath(name).name,
                label=class_to_label[class_name],
            )
            for class_name, name in train_items
        ),
        test_records=tuple(
            ImageRecord(path=name, name=PurePosixPath(name).name, label=None)
            for name in test_names
        ),
        zip_path=zip_path,
        template_member=template_member,
    )


def load_plant_data(data_path: str | Path) -> PlantData:
    """按目录结构发现数据；支持 ZIP 中额外包裹一层或多层文件夹。"""

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"数据路径不存在：{path}")
    if path.is_file() and path.suffix.casefold() == ".zip":
        return _load_zip_data(path)
    if path.is_dir():
        return _load_directory_data(path)
    raise ValueError(f"--data 必须是 ZIP 文件或数据目录：{path}")


def stratified_split_indices(
    labels: Sequence[int], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """按类别分层划分训练/验证集，避免小数据集随机切分漏掉类别。"""

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


def read_template_ids(
    data: PlantData, template_path: str | Path | None = None
) -> list[str] | None:
    """只读取模板 ID 列；即使 Category 已有内容，也绝不读取其值。"""

    archive_handle: zipfile.ZipFile | None = None
    if template_path is not None:
        path = Path(template_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"提交模板不存在：{path}")
        stream: BinaryIO = path.open("rb")
    elif data.template_path is not None:
        stream = data.template_path.open("rb")
    elif data.zip_path is not None and data.template_member is not None:
        archive_handle = zipfile.ZipFile(data.zip_path, "r")
        stream = archive_handle.open(data.template_member, "r")
    else:
        return None

    try:
        text_stream = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text_stream)
        if not reader.fieldnames:
            raise ValueError("提交模板缺少表头")
        id_field = next(
            (field for field in reader.fieldnames if field.strip().casefold() == "id"),
            None,
        )
        if id_field is None:
            raise ValueError("提交模板必须包含 ID 列")

        identifiers: list[str] = []
        for row in reader:
            identifier = (row.get(id_field) or "").strip()
            if not identifier:
                raise ValueError("提交模板中存在空 ID")
            identifiers.append(identifier)
        if not identifiers:
            raise ValueError("提交模板中没有测试图片 ID")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("提交模板中的 ID 有重复")
        return identifiers
    finally:
        # TextIOWrapper 会关闭底层流，同时也会关闭 ZIP 内部流。
        stream.close()
        if archive_handle is not None:
            archive_handle.close()


def order_test_records(
    test_records: Sequence[ImageRecord], identifiers: Sequence[str] | None
) -> tuple[list[ImageRecord], list[str]]:
    """按模板 ID 顺序组织测试图片，并检查 ID 与测试文件一一对应。"""

    if identifiers is None:
        ordered = sorted(test_records, key=lambda record: record.name.casefold())
        return ordered, [record.name for record in ordered]

    by_name = {record.name: record for record in test_records}
    if len(by_name) != len(test_records):
        raise ValueError("测试集中存在重名图片，无法用 ID 唯一匹配")

    folded_names = {name.casefold(): record for name, record in by_name.items()}
    ordered: list[ImageRecord] = []
    for identifier in identifiers:
        filename = PurePosixPath(identifier.replace("\\", "/")).name
        record = by_name.get(filename) or folded_names.get(filename.casefold())
        if record is None:
            raise ValueError(f"提交模板 ID 在测试集中不存在：{identifier}")
        ordered.append(record)

    if len(ordered) != len(test_records):
        raise ValueError(
            f"模板 ID 数量 ({len(ordered)}) 与测试图片数量 ({len(test_records)}) 不一致"
        )
    return ordered, list(identifiers)
