import argparse
import os
import signal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).parent.absolute()
FAULT_ROOT = ROOT / "data" / "download" / "fault_sticks" / "Fault_Sticks" / "data"
OUT_ROOT = ROOT / "outputs" / "fault_sticks"


def iter_fault_files(root: Path = FAULT_ROOT):
    for path in sorted(root.glob("fault*")):
        if path.is_file() and path.suffix != ".xml":
            yield path


def find_fault_file(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    matches = [path for path in iter_fault_files() if path.name == name_or_path]
    if not matches:
        needle = name_or_path.lower()
        matches = [path for path in iter_fault_files() if needle in path.name.lower()]

    if not matches:
        raise FileNotFoundError(f"No fault stick file found for: {name_or_path}")
    if len(matches) > 1:
        names = "\n".join(path.name for path in matches)
        raise ValueError(f"Multiple matching files found. Be more specific:\n{names}")
    return matches[0]


def load_fault_sticks(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                rows.append(
                    (
                        int(parts[1]),
                        int(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        parts[6],
                        int(parts[7]),
                        line_no,
                    )
                )
            except ValueError:
                continue

    dtype = [
        ("inline", "i4"),
        ("crossline", "i4"),
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
        ("fault", "U128"),
        ("stick", "i4"),
        ("line_no", "i4"),
    ]
    return np.asarray(rows, dtype=dtype)


def color_for_faults(faults: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    unique = np.unique(faults)
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    return {fault: cmap(i % cmap.N) for i, fault in enumerate(unique)}


def grouped_sticks(data: np.ndarray):
    keys = sorted(set(zip(data["fault"], data["stick"])), key=lambda x: (x[0], x[1]))
    for fault, stick in keys:
        group = data[(data["fault"] == fault) & (data["stick"] == stick)]
        group = np.sort(group, order=["z", "inline", "crossline"])
        yield fault, stick, group


def print_info(path: Path, data: np.ndarray) -> None:
    valid_inline = data["inline"] < 1_000_000_000
    valid_crossline = data["crossline"] < 1_000_000_000
    print(path.name)
    print(f"  points: {len(data)}")
    print(f"  faults: {len(np.unique(data['fault']))}")
    print(f"  sticks: {len(set(zip(data['fault'], data['stick'])))}")
    if np.any(valid_inline):
        print(f"  inline: {data['inline'][valid_inline].min()}..{data['inline'][valid_inline].max()}")
    if np.any(valid_crossline):
        print(f"  crossline: {data['crossline'][valid_crossline].min()}..{data['crossline'][valid_crossline].max()}")
    print(f"  x: {data['x'].min():.2f}..{data['x'].max():.2f}")
    print(f"  y: {data['y'].min():.2f}..{data['y'].max():.2f}")
    print(f"  z: {data['z'].min():.2f}..{data['z'].max():.2f}")
    for fault in np.unique(data["fault"])[:20]:
        count = np.count_nonzero(data["fault"] == fault)
        sticks = len(set(data["stick"][data["fault"] == fault]))
        print(f"  {fault}: {count} points, {sticks} sticks")


def plot_overview(path: Path, data: np.ndarray, output: Path | None, max_legend: int) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = output or OUT_ROOT / f"{path.name}_overview.png"
    colors = color_for_faults(data["fault"])

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    ax_map = fig.add_subplot(2, 2, 1)
    ax_inline = fig.add_subplot(2, 2, 2)
    ax_crossline = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")

    for fault, stick, group in grouped_sticks(data):
        color = colors[fault]
        label = fault if stick == min(data["stick"][data["fault"] == fault]) else None
        ax_map.plot(group["x"], group["y"], color=color, linewidth=0.8, alpha=0.85, label=label)
        valid_inline = group["inline"] < 1_000_000_000
        valid_crossline = group["crossline"] < 1_000_000_000
        if np.count_nonzero(valid_inline) > 1:
            ax_inline.plot(group["inline"][valid_inline], group["z"][valid_inline], color=color, linewidth=0.8, alpha=0.85)
        if np.count_nonzero(valid_crossline) > 1:
            ax_crossline.plot(
                group["crossline"][valid_crossline],
                group["z"][valid_crossline],
                color=color,
                linewidth=0.8,
                alpha=0.85,
            )
        ax_3d.plot(group["x"], group["y"], -group["z"], color=color, linewidth=0.7, alpha=0.8)

    ax_map.set_title("Map View")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    ax_map.set_aspect("equal", adjustable="box")

    ax_inline.set_title("Inline vs Z")
    ax_inline.set_xlabel("Inline")
    ax_inline.set_ylabel("Z")
    ax_inline.invert_yaxis()

    ax_crossline.set_title("Crossline vs Z")
    ax_crossline.set_xlabel("Crossline")
    ax_crossline.set_ylabel("Z")
    ax_crossline.invert_yaxis()

    ax_3d.set_title("3D Fault Sticks")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("-Z")
    ax_3d.view_init(elev=25, azim=-55)

    handles, labels = ax_map.get_legend_handles_labels()
    if handles:
        ax_map.legend(handles[:max_legend], labels[:max_legend], fontsize=7, loc="best")

    fig.suptitle(f"{path.name} - {len(data)} points, {len(np.unique(data['fault']))} faults")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def launch_cigvis(data: np.ndarray, port: int, line_width: float) -> None:
    from cigvis import viserplot

    mins = np.array([data["x"].min(), data["y"].min(), (-data["z"]).min()], dtype=np.float32)
    maxs = np.array([data["x"].max(), data["y"].max(), (-data["z"]).max()], dtype=np.float32)
    span = np.maximum(maxs - mins, 1.0)

    anchor = np.zeros((100, 100, 100), dtype=np.float32)
    nodes = viserplot.create_slices(
        anchor,
        clim=[-1, 1],
        cmap="gray",
        intersection_lines=False,
    )
    for _, _, group in grouped_sticks(data):
        if len(group) < 2:
            continue
        points = np.column_stack([group["x"], group["y"], -group["z"]]).astype(np.float32)
        points = (points - mins) / span * 100.0
        nodes.append(viserplot.LogLineSegments(points, line_width=line_width))

    server = viserplot.create_server(port)
    print(f"open: http://127.0.0.1:{port}")
    viserplot.plot3D(nodes, server=server)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Smeaheia fault stick interpretation files.")
    parser.add_argument("fault_file", nargs="?", help="Fault stick file name/path, e.g. GN1101 or 2010")
    parser.add_argument("--list", action="store_true", help="List available fault stick files.")
    parser.add_argument("--info", action="store_true", help="Print file summary.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--cigvis", action="store_true", help="Launch CIGVis browser 3D fault-stick view.")
    parser.add_argument("--port", type=int, default=8082, help="CIGVis browser server port.")
    parser.add_argument("--line-width", type=float, default=2.0, help="CIGVis line width.")
    parser.add_argument("--max-legend", type=int, default=15, help="Maximum legend entries in PNG overview.")
    args = parser.parse_args()

    if args.list:
        for path in iter_fault_files():
            print(path.name)
        return

    if not args.fault_file:
        parser.error("provide a fault stick file name/path, or use --list")

    path = find_fault_file(args.fault_file)
    data = load_fault_sticks(path)
    if len(data) == 0:
        raise ValueError(f"No parseable fault stick rows found in {path}")

    if args.info:
        print_info(path, data)
        return

    if args.cigvis:
        launch_cigvis(data, args.port, args.line_width)
        return

    output = plot_overview(path, data, args.output, args.max_legend)
    try:
        print(output.resolve().relative_to(ROOT))
    except ValueError:
        print(output)


if __name__ == "__main__":
    main()
