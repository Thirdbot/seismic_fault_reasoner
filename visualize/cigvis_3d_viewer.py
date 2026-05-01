import argparse
import os
import signal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import numpy as np
import segyio
from cigvis import viserplot

from seismic_3d_viewer import ROOT, find_survey


CACHE_ROOT = ROOT / "outputs" / "cigvis_cache"


def cache_path(survey_path: Path, step_inline: int, step_crossline: int, step_sample: int) -> Path:
    name = survey_path.name.replace("/", "_")
    return CACHE_ROOT / f"{name}_i{step_inline}_x{step_crossline}_t{step_sample}.npy"


def build_preview_volume(
    survey_path: Path,
    step_inline: int,
    step_crossline: int,
    step_sample: int,
    refresh: bool,
) -> tuple[np.ndarray, Path]:
    output = cache_path(survey_path, step_inline, step_crossline, step_sample)
    if output.exists() and not refresh:
        return np.load(output, mmap_mode="r"), output

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with segyio.open(str(survey_path), "r", strict=True) as segy:
        ilines = np.asarray(segy.ilines)
        xlines = np.asarray(segy.xlines)
        samples = np.asarray(segy.samples)
        selected_ilines = ilines[::step_inline]
        selected_xline_idx = np.arange(0, len(xlines), step_crossline)
        selected_sample_idx = np.arange(0, len(samples), step_sample)

        volume = np.empty(
            (len(selected_ilines), len(selected_xline_idx), len(selected_sample_idx)),
            dtype=np.float32,
        )

        for i, iline in enumerate(selected_ilines):
            section = np.asarray(segy.iline[int(iline)], dtype=np.float32)
            volume[i] = section[selected_xline_idx][:, selected_sample_idx]
            if i % 10 == 0 or i == len(selected_ilines) - 1:
                print(f"loaded inline {i + 1}/{len(selected_ilines)}", flush=True)

    np.nan_to_num(volume, copy=False)
    np.save(output, volume)
    return volume, output


def robust_clim(volume: np.ndarray, percentile: float) -> list[float]:
    sample = np.asarray(volume)
    clip = np.percentile(np.abs(sample), percentile)
    if not np.isfinite(clip) or clip == 0:
        clip = float(np.max(np.abs(sample))) or 1.0
    return [-float(clip), float(clip)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CIGVis browser viewer for Smeaheia 3D seismic.")
    parser.add_argument("survey", help="Survey name/path, e.g. GN1101 or TNE01")
    parser.add_argument("--step-inline", type=int, default=4, help="Inline decimation step.")
    parser.add_argument("--step-crossline", type=int, default=8, help="Crossline decimation step.")
    parser.add_argument("--step-sample", type=int, default=2, help="Time/depth sample decimation step.")
    parser.add_argument("--clip", type=float, default=99.0, help="Amplitude clip percentile.")
    parser.add_argument("--cmap", default="gray", help="CIGVis colormap.")
    parser.add_argument("--port", type=int, default=8080, help="Browser server port.")
    parser.add_argument("--refresh", action="store_true", help="Rebuild cached preview volume.")
    args = parser.parse_args()

    survey_path = find_survey(args.survey)
    volume, output = build_preview_volume(
        survey_path,
        args.step_inline,
        args.step_crossline,
        args.step_sample,
        args.refresh,
    )
    volume = np.asarray(volume)
    print(f"preview volume: {volume.shape} {volume.dtype}")
    print(f"cache: {output.relative_to(ROOT)}")

    nodes = viserplot.create_slices(
        volume,
        clim=robust_clim(volume, args.clip),
        cmap=args.cmap,
        intersection_lines=True,
    )
    server = viserplot.create_server(args.port)
    print(f"open: http://127.0.0.1:{args.port}")
    viserplot.plot3D(nodes, server=server)


if __name__ == "__main__":
    main()
