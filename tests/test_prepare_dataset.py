import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_dataset import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_prepares_images_and_consistent_attribute_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for punk_id in (1, 2):
                Image.new("RGB", (24, 24), (punk_id, 20, 30)).save(images / f"cryptopunk{punk_id}.jpg")

            metadata = root / "transactions.jsonl"
            records = [
                {"punk_id": 2, "type": ["Male"], "accessories": ["Earring"]},
                {"punk_id": 1, "type": ["Female"], "accessories": []},
                {"punk_id": 2, "type": [], "accessories": ["Earring"]},
                {"punk_id": 3, "type": ["Zombie"], "accessories": []},
            ]
            metadata.write_text("\n".join(json.dumps(record) for record in records))
            output = root / "pngs"
            csv_file = root / "attributes.csv"

            assert prepare_dataset(metadata, images, output, csv_file) == 2
            labels = json.loads((output / "dataset.json").read_text())["labels"]
            with csv_file.open(newline="") as source:
                rows = list(csv.reader(source))
            self.assertEqual(rows[0], ["punk_id", "filename", "Female", "Male", "Earring"])
            self.assertEqual(labels, [
                ["cryptopunk1.png", [1, 0, 0]],
                ["cryptopunk2.png", [0, 1, 1]],
            ])
            with Image.open(output / "cryptopunk1.png") as prepared:
                self.assertEqual(prepared.size, (32, 32))


if __name__ == "__main__":
    unittest.main()
