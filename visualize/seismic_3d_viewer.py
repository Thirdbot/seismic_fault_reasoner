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
SURVEY_ROOT = ROOT / "data" / "download" / "seismic_3d_surveys" / "Seismic_3D_Surveys" / "data"
OUT_ROOT = ROOT / "outputs" / "seismic_3d"


def iter_surveys(root: Path = SURVEY_ROOT):
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix not in {".xml", ".pdf"}:
            yield path


def find_survey(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    matches = [path for path in iter_surveys() if path.name == name_or_path]
    if not matches:
        needle = name_or_path.lower()
        matches = [path for path in iter_surveys() if needle in path.name.lower()]

    if not matches:
        raise FileNotFoundError(f"No 3D survey found for: {name_or_path}")
    if len(matches) > 1:
        names = "\n".join(path.name for path in matches)
        raise ValueError(f"Multiple matching surveys found. Be more specific:\n{names}")
    return matches[0]


def nearest(values: np.ndarray, requested: int | None) -> int:
    if requested is None:
        return int(values[len(values) // 2])
    return int(values[np.abs(values - requested).argmin()])


def normalize_for_display(data: np.ndarray, clip_percentile: float) -> tuple[np.ndarray, float]:
    data = np.nan_to_num(data.astype(np.float32, copy=False), copy=False)
    clip = np.percentile(np.abs(data), clip_percentile)
    if not np.isfinite(clip) or clip == 0:
        clip = float(np.max(np.abs(data))) or 1.0
    return np.clip(data, -clip, clip), clip


def read_slice(path: Path, mode: str, index: int | None):
    with segyio.open(str(path), "r", strict=True) as segy:
        ilines = np.asarray(segy.ilines)
        xlines = np.asarray(segy.xlines)
        samples = np.asarray(segy.samples)
        dt_us = float(segyio.tools.dt(segy))

        if mode == "inline":
            selected = nearest(ilines, index)
            data = np.asarray(segy.iline[selected], dtype=np.float32)
            x_label = "Crossline"
            y_label = "Time/depth sample"
            extent = [int(xlines[0]), int(xlines[-1]), float(samples[-1]), float(samples[0])]
        elif mode == "crossline":
            selected = nearest(xlines, index)
            data = np.asarray(segy.xline[selected], dtype=np.float32)
            x_label = "Inline"
            y_label = "Time/depth sample"
            extent = [int(ilines[0]), int(ilines[-1]), float(samples[-1]), float(samples[0])]
        elif mode == "timeslice":
            selected = nearest(samples, index)
            sample_idx = int(np.abs(samples - selected).argmin())
            data = np.asarray(segy.depth_slice[sample_idx], dtype=np.float32)
            x_label = "Crossline"
            y_label = "Inline"
            extent = [int(xlines[0]), int(xlines[-1]), int(ilines[-1]), int(ilines[0])]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return data, selected, dt_us, extent, x_label, y_label


def plot_slice(
    path: Path,
    mode: str,
    index: int | None,
    output: Path | None,
    clip_percentile: float,
    cmap: str,
    dpi: int,
) -> Path:
    data, selected, dt_us, extent, x_label, y_label = read_slice(path, mode, index)
    display, clip = normalize_for_display(data, clip_percentile)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = output or OUT_ROOT / f"{path.name}_{mode}_{selected}.png"

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = ax.imshow(
        display.T if mode in {"inline", "crossline"} else display,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        vmin=-clip,
        vmax=clip,
    )
    ax.set_title(f"{path.name} {mode} {selected}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.text(
        0.01,
        0.01,
        f"shape={data.shape} dt={dt_us:g} us clip=p{clip_percentile:g}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4},
    )
    fig.colorbar(image, ax=ax, label="Amplitude")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def print_info(path: Path) -> None:
    with segyio.open(str(path), "r", strict=True) as segy:
        print(path.name)
        print(f"  traces: {segy.tracecount}")
        print(f"  samples: {len(segy.samples)} [{segy.samples[0]}..{segy.samples[-1]}]")
        print(f"  dt_us: {segyio.tools.dt(segy):g}")
        print(f"  inlines: {len(segy.ilines)} [{segy.ilines[0]}..{segy.ilines[-1]}]")
        print(f"  crosslines: {len(segy.xlines)} [{segy.xlines[0]}..{segy.xlines[-1]}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Smeaheia 3D seismic survey slices.")
    parser.add_argument("survey", nargs="?", help="Survey name/path, e.g. GN1101 or TNE01")
    parser.add_argument("--list", action="store_true", help="List available 3D surveys.")
    parser.add_argument("--info", action="store_true", help="Print survey geometry summary.")
    parser.add_argument(
        "--mode",
        choices=("inline", "crossline", "timeslice"),
        default="inline",
        help="Slice orientation.",
    )
    parser.add_argument("--index", type=int, help="Requested inline, crossline, or sample/time value.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--clip", type=float, default=99.0, help="Amplitude clip percentile.")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    args = parser.parse_args()

    if args.list:
        for path in iter_surveys():
            print(path.name)
        return

    if not args.survey:
        parser.error("provide a survey name/path, or use --list")

    survey_path = find_survey(args.survey)
    if args.info:
        print_info(survey_path)
        return

    output = plot_slice(
        survey_path,
        args.mode,
        args.index,
        args.output,
        args.clip,
        args.cmap,
        args.dpi,
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
