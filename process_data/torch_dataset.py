import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FaultDetectionDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "process_data/fault_detection/data",
        info_file: str = "info.jsonl",
        records: Optional[list[dict]] = None,
        transform: Optional[Callable] = None,
        crop_size: Optional[tuple[int, int]] = None,
        center_crop: bool = False,
        resize: Optional[tuple[int, int]] = None,
        mmap: bool = False,
    ):
        self.root = Path(root)
        self.info_path = self.root / info_file
        self.transform = transform
        self.crop_size = crop_size
        self.center_crop = center_crop
        self.resize = resize
        self.mmap = mmap

        if records is None:
            if not self.info_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.info_path}")
            records = read_jsonl(self.info_path)

        if not records:
            raise ValueError(f"No fault-detection records found in {self.info_path}")

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = self._load_npy(record["image_path"], dtype=np.float32)
        mask = self._load_npy(record["mask_path"], dtype=np.float32)

        image, mask = self._crop(image, mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = self._to_tensor(image)
        mask = self._to_tensor(mask)

        if self.resize is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=self.resize,
                mode="nearest",
            ).squeeze(0)

        return {
            "image": image,
            "mask": (mask > 0).float(),
            "metadata": record,
        }

    def _load_npy(self, path: str, dtype) -> np.ndarray:
        array = np.load(path, mmap_mode="r" if self.mmap else None)
        return np.asarray(array, dtype=dtype)

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_size is None:
            return image, mask

        crop_h, crop_w = self.crop_size
        height, width = image.shape
        if crop_h > height or crop_w > width:
            raise ValueError(f"crop_size {self.crop_size} is larger than image shape {image.shape}")

        if self.center_crop:
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2
        else:
            top = np.random.randint(0, height - crop_h + 1)
            left = np.random.randint(0, width - crop_w + 1)

        return image[top:top + crop_h, left:left + crop_w], mask[top:top + crop_h, left:left + crop_w]

    @staticmethod
    def _to_tensor(array) -> torch.Tensor:
        if torch.is_tensor(array):
            tensor = array.float()
        else:
            tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.contiguous()


class InterpretationDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "process_data/interpretation",
        records: Optional[list[dict]] = None,
        include_text_chunks: bool = True,
        include_media: bool = True,
        load_images: bool = True,
        image_size: Optional[tuple[int, int]] = None,
    ):
        self.root = Path(root)
        self.include_text_chunks = include_text_chunks
        self.include_media = include_media
        self.load_images = load_images
        self.image_size = image_size

        self.records = records if records is not None else collect_interpretation_records(
            self.root,
            include_text_chunks=include_text_chunks,
            include_media=include_media,
        )
        if not self.records:
            raise ValueError(f"No interpretation records found in {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        item = {
            "kind": record["kind"],
            "doc_id": record.get("doc_id"),
            "text": record.get("text", ""),
            "metadata": record,
        }

        image_path = record.get("image_path")
        if image_path:
            item["image_path"] = image_path
            item["image"] = self._load_image(image_path) if self.load_images else None
        else:
            item["image_path"] = None
            item["image"] = None

        return item

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load interpretation images") from exc

        image = Image.open(path).convert("RGB")
        if self.image_size is not None:
            image = image.resize((self.image_size[1], self.image_size[0]))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_interpretation_records(
    root: str | Path,
    include_text_chunks: bool = True,
    include_media: bool = True,
) -> list[dict]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Interpretation folder not found: {root}")

    records: list[dict] = []
    doc_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "extracted_data")

    for doc_dir in doc_dirs:
        if include_media:
            manifest_path = doc_dir / "manifest.jsonl"
            if manifest_path.exists():
                for record in read_jsonl(manifest_path):
                    caption = record.get("caption", "")
                    context = record.get("same_page_text", "")
                    records.append({
                        **record,
                        "kind": "media",
                        "text": caption if not context else f"{caption}\n{context}",
                    })

        if include_text_chunks:
            chunks_path = doc_dir / "chunks.jsonl"
            if chunks_path.exists():
                for record in read_jsonl(chunks_path):
                    records.append({
                        **record,
                        "kind": "text",
                        "image_path": None,
                    })

    return records


def save_dataset_payload(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def build_fault_detection_dataset(
    root: str | Path,
    output_path: str | Path,
    info_file: str = "info.jsonl",
) -> Path:
    dataset = FaultDetectionDataset(root=root, info_file=info_file, mmap=True)
    payload = {
        "dataset_type": "fault_detection",
        "root": str(Path(root).resolve()),
        "info_file": info_file,
        "records": dataset.records,
    }
    return save_dataset_payload(output_path, payload)


def build_interpretation_dataset(
    root: str | Path,
    output_path: str | Path,
    include_text_chunks: bool = True,
    include_media: bool = True,
) -> Path:
    records = collect_interpretation_records(
        root,
        include_text_chunks=include_text_chunks,
        include_media=include_media,
    )
    if not records:
        raise ValueError(f"No interpretation records found in {root}")

    payload = {
        "dataset_type": "interpretation",
        "root": str(Path(root).resolve()),
        "include_text_chunks": include_text_chunks,
        "include_media": include_media,
        "records": records,
    }
    return save_dataset_payload(output_path, payload)


def load_saved_dataset(path: str | Path, **dataset_kwargs) -> Dataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dataset_type = payload.get("dataset_type")

    if dataset_type == "fault_detection":
        return FaultDetectionDataset(
            root=payload["root"],
            info_file=payload.get("info_file", "info.jsonl"),
            records=payload["records"],
            **dataset_kwargs,
        )
    if dataset_type == "interpretation":
        return InterpretationDataset(
            root=payload["root"],
            records=payload["records"],
            **dataset_kwargs,
        )

    raise ValueError(f"Unknown dataset_type in {path}: {dataset_type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build torch-loadable dataset payloads from process_data folders."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fault-only", action="store_true")
    parser.add_argument("--interpretation-only", action="store_true")
    parser.add_argument("--no-text-chunks", action="store_true")
    parser.add_argument("--no-media", action="store_true")
    args = parser.parse_args()

    process_root = args.root
    output_dir = args.output_dir or process_root / "datasets"
    build_fault = not args.interpretation_only
    build_interpretation = not args.fault_only

    if build_fault:
        path = build_fault_detection_dataset(
            root=process_root / "fault_detection" / "data",
            output_path=output_dir / "fault_detection_dataset.pt",
        )
        print(f"saved fault_detection dataset: {path}")

    if build_interpretation:
        path = build_interpretation_dataset(
            root=process_root / "interpretation",
            output_path=output_dir / "interpretation_dataset.pt",
            include_text_chunks=not args.no_text_chunks,
            include_media=not args.no_media,
        )
        print(f"saved interpretation dataset: {path}")


if __name__ == "__main__":
    main()
