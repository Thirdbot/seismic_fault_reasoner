import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


LOSS_KEYS = ("loss", "text_loss", "seg_loss")
SCORE_KEYS = ("seg_dice", "seg_iou", "seg_dice_positive", "seg_iou_positive")
RATIO_KEYS = ("seg_pred_positive_ratio", "seg_target_positive_ratio", "seg_positive_mask_ratio")


def read_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def select_rows(records: list[dict], split: str) -> list[dict]:
    return [record for record in records if record.get("type") == split]


def aggregate_by_epoch(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        grouped.setdefault(int(epoch), []).append(row)

    aggregated = []
    for epoch, epoch_rows in sorted(grouped.items()):
        item = {"epoch": epoch}
        for key in keys:
            values = [row[key] for row in epoch_rows if row.get(key) is not None]
            if values:
                item[key] = sum(values) / len(values)
        aggregated.append(item)
    return aggregated


def plot_series(
    rows: list[dict],
    keys: tuple[str, ...],
    title: str,
    output_path: Path,
    x_axis: str,
) -> bool:
    if x_axis == "epoch":
        rows = aggregate_by_epoch(rows, keys)
    rows = [row for row in rows if row.get(x_axis) is not None]
    if not rows:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plotted = False
    for key in keys:
        xs = [row[x_axis] for row in rows if row.get(key) is not None]
        ys = [row[key] for row in rows if row.get(key) is not None]
        if ys:
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=key)
            plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title(title)
    plt.xlabel(x_axis)
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
    parser.add_argument("--x-axis", choices=["epoch", "step"], default="epoch")
    args = parser.parse_args()

    records = read_metrics(args.metrics)
    train_rows = select_rows(records, "train")
    val_rows = select_rows(records, "val")

    outputs = []
    suffix = "" if args.x_axis == "epoch" else "_by_step"
    train_loss_path = args.output_dir / f"train_loss{suffix}.png"
    val_loss_path = args.output_dir / f"val_loss{suffix}.png"
    val_segmentation_path = args.output_dir / f"val_segmentation{suffix}.png"
    val_ratios_path = args.output_dir / f"val_segmentation_ratios{suffix}.png"
    if plot_series(train_rows, LOSS_KEYS, "Training Loss", train_loss_path, args.x_axis):
        outputs.append(train_loss_path)
    if plot_series(val_rows, LOSS_KEYS, "Validation Loss", val_loss_path, args.x_axis):
        outputs.append(val_loss_path)
    if plot_series(val_rows, SCORE_KEYS, "Validation Segmentation Metrics", val_segmentation_path, args.x_axis):
        outputs.append(val_segmentation_path)
    if plot_series(val_rows, RATIO_KEYS, "Validation Segmentation Ratios", val_ratios_path, args.x_axis):
        outputs.append(val_ratios_path)

    if not outputs:
        raise ValueError(f"No plottable train/val rows found in {args.metrics}")

    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
