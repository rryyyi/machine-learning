"""读取提交模板 ID 并保证预测结果按模板顺序排列。"""

import csv
import io
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence

from .dataset import ImageRecord, PlantData


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

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("提交模板中的 ID 有重复")

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
