import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


LOSS_KEYS = ("loss", "text_loss", "seg_loss")
SCORE_KEYS = ("seg_dice", "seg_iou")


def read_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def select_rows(records: list[dict], split: str) -> list[dict]:
    return [record for record in records if record.get("type") == split]


def plot_series(rows: list[dict], keys: tuple[str, ...], title: str, output_path: Path) -> bool:
    rows = [row for row in rows if row.get("step") is not None]
    if not rows:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plotted = False
    for key in keys:
        xs = [row["step"] for row in rows if row.get(key) is not None]
        ys = [row[key] for row in rows if row.get(key) is not None]
        if ys:
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=key)
            plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title(title)
    plt.xlabel("step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics JSONL as PNG charts.")
    parser.add_argument("--metrics", type=Path, default=Path("results/multitask/metrics.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/multitask"))
    args = parser.parse_args()

    records = read_metrics(args.metrics)
    train_rows = select_rows(records, "train")
    val_rows = select_rows(records, "val")

    outputs = []
    if plot_series(train_rows, LOSS_KEYS, "Training Loss", args.output_dir / "train_loss.png"):
        outputs.append(args.output_dir / "train_loss.png")
    if plot_series(val_rows, LOSS_KEYS, "Validation Loss", args.output_dir / "val_loss.png"):
        outputs.append(args.output_dir / "val_loss.png")
    if plot_series(val_rows, SCORE_KEYS, "Validation Segmentation Metrics", args.output_dir / "val_segmentation.png"):
        outputs.append(args.output_dir / "val_segmentation.png")

    if not outputs:
        raise ValueError(f"No plottable train/val rows found in {args.metrics}")

    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
