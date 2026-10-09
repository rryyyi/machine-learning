"""植物分类实验的数据发现、读取和提交文件处理。"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

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
