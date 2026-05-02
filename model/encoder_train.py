import argparse
import sys
from pathlib import Path

home_path = Path(__file__).parent.parent.absolute()
if home_path.as_posix() not in sys.path:
    sys.path.insert(0, home_path.as_posix())

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from process_data import FaultDetectionDataset


def split_dataset(dataset, val_fraction, seed):
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def main():
    home_path = Path(__file__).parent.parent.absolute()
    output_dir = home_path / "outputs" / "fault_encoder"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = home_path / "process_data" / "fault_detection" / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_height,crop_width = (128,128)
    val_fraction = 0.2
    batch_size = 1
    num_workers = 0
    seed = 42

    torch.manual_seed(seed)


    dataset = FaultDetectionDataset(
        root= datasets_dir.as_posix(),
        crop_size=(crop_height,crop_width),
        center_crop=False,
        mmap=True,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_fraction, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(next(iter(train_loader)))

if __name__ == "__main__":
    main()

