import argparse
import json
import os
import signal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFO = ROOT / "process_data" / "fault_detection" / "data" / "info.jsonl"
DEFAULT_OUT = ROOT / "outputs" / "fault_detection"


def load_records(info_path: Path) -> list[dict]:
    if not info_path.exists():
        raise FileNotFoundError(f"metadata file not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_record(records: list[dict], index: int | None, inline: int | None) -> dict:
    if inline is not None:
        matches = [record for record in records if record.get("slice_index") == inline]
        if not matches:
            raise ValueError(f"inline {inline} not found in metadata")
        return matches[0]

    index = 0 if index is None else index
    if index < 0 or index >= len(records):
        raise IndexError(f"index {index} out of range for {len(records)} records")
    return records[index]


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    base = np.clip((image + 1.0) / 2.0, 0.0, 1.0)
    rgb = np.stack([base, base, base], axis=-1)
    fault = mask > 0
    rgb[fault, 0] = (1.0 - alpha) * rgb[fault, 0] + alpha
    rgb[fault, 1] = (1.0 - alpha) * rgb[fault, 1]
    rgb[fault, 2] = (1.0 - alpha) * rgb[fault, 2]
    return rgb


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to resize predicted masks") from exc

    resized = Image.fromarray(mask.astype(np.float32)).resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.float32)


def visualize_record(record: dict, output: Path, alpha: float, dpi: int, mask_path: Path | None = None) -> Path:
    image = np.load(record["image_path"]).astype(np.float32)
    selected_mask_path = mask_path or Path(record["mask_path"])
    mask = np.load(selected_mask_path).astype(np.float32)
    mask = resize_mask(mask, image.shape)
    overlay = make_overlay(image, mask, alpha)

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("Original seismic")

    axes[1].imshow(mask, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Fault mask" if mask_path is None else "Predicted fault mask")

    axes[2].imshow(overlay, aspect="auto")
    axes[2].set_title("Mask overlay")

    for ax in axes:
        ax.set_xlabel("Crossline pixel")
        ax.set_ylabel("Sample pixel")

    title = f"{record.get('survey', 'survey')} {record.get('slice_type', 'slice')} {record.get('slice_index')}"
    fault_names = record.get("fault_names", [])
    if fault_names:
        title += f" | faults: {', '.join(fault_names[:5])}"
        if len(fault_names) > 5:
            title += f" +{len(fault_names) - 5}"
    fig.suptitle(title)

    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize seismic image and fault-detection mask.")
    parser.add_argument("--info", type=Path, default=DEFAULT_INFO, help="Path to info.jsonl.")
    parser.add_argument("--index", type=int, help="Metadata row index to visualize.")
    parser.add_argument("--inline", type=int, help="Inline id to visualize.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--mask", type=Path, help="Optional predicted mask .npy path to overlay instead of ground truth.")
    parser.add_argument("--alpha", type=float, default=0.75, help="Mask overlay alpha.")
    parser.add_argument("--dpi", type=int, default=160, help="Output PNG DPI.")
    args = parser.parse_args()

    records = load_records(args.info)
    record = select_record(records, args.index, args.inline)
    output = args.output or DEFAULT_OUT / f"{record['survey']}_{record['slice_type']}_{record['slice_index']}_mask_overlay.png"
    output = visualize_record(record, output, args.alpha, args.dpi, args.mask)
    print(output.resolve().relative_to(ROOT))


if __name__ == "__main__":
    main()
