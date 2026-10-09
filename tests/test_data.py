import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from plant_classifier.data import (
    ImageRecord,
    PlantImageDataset,
    load_plant_data,
    order_test_records,
    read_template_ids,
    stratified_split_indices,
)


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(buffer, format="PNG")
    return buffer.getvalue()


class PlantDataTests(unittest.TestCase):
    def test_stratified_split_preserves_each_class(self):
        labels = [0] * 10 + [1] * 10 + [2] * 10
        train_indices, val_indices = stratified_split_indices(labels, 0.2, seed=7)

        self.assertEqual(len(train_indices), 24)
        self.assertEqual(len(val_indices), 6)
        self.assertTrue(set(train_indices).isdisjoint(val_indices))
        for label in set(labels):
            self.assertTrue(any(labels[index] == label for index in train_indices))
            self.assertTrue(any(labels[index] == label for index in val_indices))

    def test_zip_dataset_ignores_existing_category_values_in_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "task1.zip"
            root = "dataset-for-task2/dataset-for-task2"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(f"{root}/train/Plant A/a.png", png_bytes((0, 255, 0)))
                archive.writestr(f"{root}/train/Plant A/b.png", png_bytes((0, 200, 0)))
                archive.writestr(f"{root}/train/Plant B/c.png", png_bytes((255, 0, 0)))
                archive.writestr(f"{root}/train/Plant B/d.png", png_bytes((200, 0, 0)))
                archive.writestr(f"{root}/test/z.png", png_bytes((0, 0, 255)))
                archive.writestr(f"{root}/test/y.png", png_bytes((0, 0, 200)))
                archive.writestr(
                    "submission-for-task2.csv",
                    "ID,Category\nz.png,LEAKED LABEL\ny.png,ANOTHER LABEL\n",
                )

            data = load_plant_data(zip_path)
            self.assertEqual(data.classes, ("Plant A", "Plant B"))
            self.assertEqual(len(data.train_records), 4)
            self.assertEqual(len(data.test_records), 2)
            self.assertEqual(read_template_ids(data), ["z.png", "y.png"])

            ordered, identifiers = order_test_records(
                data.test_records, read_template_ids(data)
            )
            self.assertEqual(identifiers, ["z.png", "y.png"])
            self.assertEqual([record.name for record in ordered], ["z.png", "y.png"])

            image, label = PlantImageDataset(
                data.train_records, zip_path=data.zip_path
            )[0]
            self.assertIsInstance(image, Image.Image)
            self.assertIn(label, (0, 1))

    def test_ordering_rejects_ids_that_do_not_match_test_images(self):
        records = [ImageRecord("/tmp/a.png", "a.png", None)]
        with self.assertRaisesRegex(ValueError, "不存在"):
            order_test_records(records, ["missing.png"])


if __name__ == "__main__":
    unittest.main()
