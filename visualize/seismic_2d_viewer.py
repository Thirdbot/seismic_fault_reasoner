import argparse
import os
import signal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import matplotlib.pyplot as plt
import numpy as np
import segyio


ROOT = Path(__file__).parent.absolute()
SEGY_ROOT = ROOT / "data" / "download" / "seismic_2d_lines" / "data" / "segy"
OUT_ROOT = ROOT / "outputs" / "seismic_2d"


def iter_segy_lines(root: Path = SEGY_ROOT):
    for path in sorted(root.glob("*/*")):
        if path.is_file() and path.suffix != ".xml":
            yield path


def find_line(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    matches = [path for path in iter_segy_lines() if path.name == name_or_path]
    if not matches:
        matches = [path for path in iter_segy_lines() if name_or_path.lower() in path.name.lower()]

    if not matches:
        raise FileNotFoundError(f"No seismic 2D line found for: {name_or_path}")
    if len(matches) > 1:
        names = "\n".join(str(path.relative_to(ROOT)) for path in matches[:20])
        raise ValueError(f"Multiple matching lines found. Be more specific:\n{names}")
    return matches[0]


def read_segy(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with segyio.open(str(path), "r", strict=False, ignore_geometry=True) as segy:
        samples = np.asarray(segy.samples, dtype=np.float32)
        data = segyio.tools.collect(segy.trace[:]).astype(np.float32)
        dt_us = float(segyio.tools.dt(segy))

    data = np.nan_to_num(data, copy=False)
    return data, samples, dt_us


def normalize_for_display(data: np.ndarray, clip_percentile: float) -> tuple[np.ndarray, float]:
    clip = np.percentile(np.abs(data), clip_percentile)
    if not np.isfinite(clip) or clip == 0:
        clip = float(np.max(np.abs(data))) or 1.0
    return np.clip(data, -clip, clip), clip


def plot_line(
    path: Path,
    output: Path | None,
    clip_percentile: float,
    cmap: str,
    dpi: int,
) -> Path:
    data, samples, dt_us = read_segy(path)
    display, clip = normalize_for_display(data, clip_percentile)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = output or OUT_ROOT / f"{path.parent.name}_{path.name}.png"

    fig_width = min(max(data.shape[0] / 120, 9), 22)
    fig, ax = plt.subplots(figsize=(fig_width, 8), constrained_layout=True)
    extent = [1, data.shape[0], float(samples[-1]), float(samples[0])]
    image = ax.imshow(
        display.T,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        vmin=-clip,
        vmax=clip,
    )

    ax.set_title(f"{path.parent.name}/{path.name}")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time/depth sample")
    ax.text(
        0.01,
        0.01,
        f"traces={data.shape[0]} samples={data.shape[1]} dt={dt_us:g} us clip=p{clip_percentile:g}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4},
    )
    fig.colorbar(image, ax=ax, label="Amplitude")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Smeaheia 2D seismic SEG-Y lines.")
    parser.add_argument("line", nargs="?", help="Line name or path, e.g. GSB-85R97-202")
    parser.add_argument("--list", action="store_true", help="List available line names.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--clip", type=float, default=99.0, help="Amplitude clip percentile.")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    args = parser.parse_args()

    if args.list:
        for path in iter_segy_lines():
            print(path.relative_to(ROOT))
        return

    if not args.line:
        parser.error("provide a line name/path, or use --list")

    line_path = find_line(args.line)
    output = plot_line(line_path, args.output, args.clip, args.cmap, args.dpi)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
