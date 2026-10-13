"""Prepare image files and attribute metadata for conditional GAN training."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def prepare_dataset(metadata: Path, images: Path, output: Path, attributes_csv: Path) -> int:
    """Return the number of images prepared for training."""
    punk_attributes = defaultdict(lambda: {"type": set(), "accessories": set()})

    with metadata.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            punk_id = int(record["punk_id"])
            if not (images / f"cryptopunk{punk_id}.jpg").is_file():
                continue
            punk_attributes[punk_id]["type"].update(record.get("type", []))
            punk_attributes[punk_id]["accessories"].update(record.get("accessories", []))

    if not punk_attributes:
        raise ValueError("No JPEG images matched the metadata's punk_id values")

    types = sorted({name for attrs in punk_attributes.values() for name in attrs["type"]})
    accessories = sorted({name for attrs in punk_attributes.values() for name in attrs["accessories"]})
    attribute_names = types + accessories

    output.mkdir(parents=True, exist_ok=True)
    attributes_csv.parent.mkdir(parents=True, exist_ok=True)
    labels = []

    with attributes_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["punk_id", "filename", *attribute_names])

        for punk_id, attrs in sorted(punk_attributes.items()):
            filename = f"cryptopunk{punk_id}.png"
            values = [int(name in attrs["type"] or name in attrs["accessories"]) for name in attribute_names]
            writer.writerow([punk_id, f"cryptopunk{punk_id}.jpg", *values])

            with Image.open(images / f"cryptopunk{punk_id}.jpg") as image:
                image.convert("RGB").resize((32, 32), Image.Resampling.NEAREST).save(output / filename)
            labels.append([filename, values])

    with (output / "dataset.json").open("w", encoding="utf-8") as destination:
        json.dump({"labels": labels}, destination)

    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/cryptopunks/transactions.jsonl"))
    parser.add_argument("--images", type=Path, default=Path("data/cryptopunks/images"))
    parser.add_argument("--output", type=Path, default=Path("data/cryptopunks/pngs"))
    parser.add_argument("--attributes-csv", type=Path, default=Path("attributes.csv"))
    args = parser.parse_args()

    count = prepare_dataset(args.metadata, args.images, args.output, args.attributes_csv)
    print(f"Prepared {count} images and their attribute vectors in {args.output}")


if __name__ == "__main__":
    main()
