import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class FaultDetectionDataset(Dataset):

    def __init__(
        self,
        root: str | Path = "process_data/fault_detection/data",
        info_file: str = "info.jsonl",
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

        if not self.info_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.info_path}")

        with self.info_path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        if not self.records:
            raise ValueError(f"No records found in {self.info_path}")

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

        mask = (mask > 0).float()

        return {
            "image": image,
            "mask": mask,
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

        image = image[top:top + crop_h, left:left + crop_w]
        mask = mask[top:top + crop_h, left:left + crop_w]
        return image, mask

    @staticmethod
    def _to_tensor(array) -> torch.Tensor:
        if torch.is_tensor(array):
            tensor = array.float()
        else:
            tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.contiguous()