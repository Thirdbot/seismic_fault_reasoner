import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import segyio


class FaultDetectionDataBuilder:
    def __init__(self, mask_radius=3):
        self.home = Path(__file__).resolve().parent.parent
        self.raw_data_path = self.home / "data" / "download"
        self.data_path = Path(__file__).resolve().parent / "fault_detection" / "data"
        self.images_path = self.data_path / "images"
        self.masks_path = self.data_path / "masks"
        self.labels_path = self.masks_path
        self.info_path = self.data_path / "info.jsonl"

        self.images_path.mkdir(parents=True, exist_ok=True)
        self.masks_path.mkdir(parents=True, exist_ok=True)

        survey_dir = self.raw_data_path / "seismic_3d_surveys" / "Seismic_3D_Surveys" / "data"
        fault_dir = self.raw_data_path / "fault_sticks" / "Fault_Sticks" / "data"
        self.segy_path = survey_dir / "GN1101_Scaled(Realized)"
        self.fault_sticks_path = fault_dir / "fault_Sticks_GN1101_2012"
        self.seg_y_GN1101_path = self.segy_path
        self.fault_sticks_GN1101_path = self.fault_sticks_path
        self.mask_radius = mask_radius

    def _read_segy(self, verbose=True):
        if not self.segy_path.exists():
            raise FileNotFoundError(f"SEG-Y file not found: {self.segy_path}")

        with segyio.open(self.segy_path.as_posix(), "r", strict=True) as segy:
            if verbose:
                print(f"file: {self.segy_path}")
                print(f"size_gb: {self.segy_path.stat().st_size / 1024 ** 3:.2f}")
                print(f"trace_count: {segy.tracecount}")
                print(f"sample_count: {len(segy.samples)}")
                print(f"sample_range: {segy.samples[0]}..{segy.samples[-1]}")
                print(f"sample_interval_us: {segyio.tools.dt(segy)}")
                print(f"inline_count: {len(segy.ilines)}")
                print(f"inline_range: {segy.ilines[0]}..{segy.ilines[-1]}")
                print(f"crossline_count: {len(segy.xlines)}")
                print(f"crossline_range: {segy.xlines[0]}..{segy.xlines[-1]}")

            return np.asarray(segy.ilines), np.asarray(segy.xlines), np.asarray(segy.samples)

    def _read_fault_sticks(self, verbose=True):
        if not self.fault_sticks_path.exists():
            raise FileNotFoundError(f"Fault sticks file not found: {self.fault_sticks_path}")

        rows = []
        with self.fault_sticks_path.open("r", encoding="latin-1", errors="replace") as file:
            for line in file:
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    rows.append({
                        "pick_type": parts[0],
                        "inline": int(parts[1]),
                        "crossline": int(parts[2]),
                        "x": float(parts[3]),
                        "y": float(parts[4]),
                        "z": float(parts[5]),
                        "fault_name": parts[6],
                        "stick_id": int(parts[7]),
                    })
                except ValueError:
                    continue

        df = pd.DataFrame(rows)
        if verbose and not df.empty:
            print(df.head())
            print(f"fault_names: {df['fault_name'].nunique()}")
            print(df[["inline", "crossline", "z"]].describe())
        return df

    @staticmethod
    def _nearest_indices(axis, values):
        axis = np.asarray(axis)
        values = np.asarray(values)
        indices = np.searchsorted(axis, values)
        indices = np.clip(indices, 1, len(axis) - 1)
        left = axis[indices - 1]
        right = axis[indices]
        return np.where(np.abs(values - left) <= np.abs(values - right), indices - 1, indices).astype(np.int64)

    @staticmethod
    def _draw_line(mask, x0, y0, x1, y1):
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        if steps <= 1:
            mask[y0, x0] = 1
            return
        xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int64)
        ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int64)
        valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
        mask[ys[valid], xs[valid]] = 1

    @staticmethod
    def _dilate_mask(mask, radius):
        if radius <= 0:
            return mask
        padded = np.pad(mask, radius, mode="constant")
        dilated = np.zeros_like(mask)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    y0 = radius + dy
                    x0 = radius + dx
                    dilated = np.maximum(dilated, padded[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]])
        return dilated.astype(np.uint8)

    @staticmethod
    def _normalize_section(section, clip_percentile=99.0):
        section = np.nan_to_num(section.astype(np.float32, copy=False), copy=False)
        clip = np.percentile(np.abs(section), clip_percentile)
        if not np.isfinite(clip) or clip == 0:
            clip = float(np.max(np.abs(section))) or 1.0
        section = np.clip(section, -clip, clip) / clip
        return section.astype(np.float32)

    def _filter_fault_sticks(self, df, ilines, xlines, samples):
        df = df.copy()
        df = df[(df["inline"] < 1_000_000_000) & (df["crossline"] < 1_000_000_000)]
        df = df[df["pick_type"] == "INLINE-"]
        df = df[df["inline"].between(int(ilines[0]), int(ilines[-1]))]
        df = df[df["crossline"].between(int(xlines[0]), int(xlines[-1]))]
        df = df[df["z"].between(float(samples[0]), float(samples[-1]))]
        return df

    def _build_inline_mask(self, points_on_inline, xlines, samples):
        mask = np.zeros((len(samples), len(xlines)), dtype=np.uint8)
        if points_on_inline.empty:
            return mask, []

        fault_names = []
        for (fault_name, stick_id), group in points_on_inline.groupby(["fault_name", "stick_id"]):
            if len(group) < 2:
                continue
            group = group.sort_values("z")
            xs = self._nearest_indices(xlines, group["crossline"].to_numpy())
            ys = self._nearest_indices(samples, group["z"].to_numpy())
            for idx in range(len(xs) - 1):
                self._draw_line(mask, int(xs[idx]), int(ys[idx]), int(xs[idx + 1]), int(ys[idx + 1]))
            fault_names.append(str(fault_name))

        return self._dilate_mask(mask, self.mask_radius), sorted(set(fault_names))

    def map_data(self, max_inlines=None, include_negative=True, keep_empty_labeled=False, clip_percentile=99.0):
        ilines, xlines, samples = self._read_segy(verbose=True)
        df = self._read_fault_sticks(verbose=True)
        df = self._filter_fault_sticks(df, ilines, xlines, samples)

        print(f"filtered_fault_points: {len(df)}")
        print(f"filtered_fault_inlines: {df['inline'].nunique()}")

        labeled_inlines = sorted(set(df["inline"].astype(int)))
        if include_negative:
            inline_ids = list(map(int, ilines))
        else:
            inline_ids = labeled_inlines

        if max_inlines is not None:
            inline_ids = inline_ids[:max_inlines]

        written = 0
        skipped_empty_labeled = 0
        with segyio.open(self.segy_path.as_posix(), "r", strict=True) as segy, self.info_path.open("w", encoding="utf-8") as info:
            for inline_id in inline_ids:
                section = np.asarray(segy.iline[int(inline_id)], dtype=np.float32).T
                image = self._normalize_section(section, clip_percentile=clip_percentile)
                points_on_inline = df[df["inline"] == inline_id]
                mask, fault_names = self._build_inline_mask(points_on_inline, xlines, samples)
                has_fault = bool(mask.any())

                if not has_fault and not include_negative and not keep_empty_labeled:
                    skipped_empty_labeled += 1
                    continue

                image_name = f"GN1101_inline_{int(inline_id)}.npy"
                mask_name = f"GN1101_inline_{int(inline_id)}_mask.npy"
                image_path = self.images_path / image_name
                mask_path = self.masks_path / mask_name
                np.save(image_path, image)
                np.save(mask_path, mask)

                info.write(json.dumps({
                    "survey": "GN1101",
                    "slice_type": "inline",
                    "slice_index": int(inline_id),
                    "image_path": image_path.as_posix(),
                    "mask_path": mask_path.as_posix(),
                    "image_shape": list(image.shape),
                    "mask_shape": list(mask.shape),
                    "has_fault": has_fault,
                    "fault_names": fault_names,
                    "fault_point_count": int(len(points_on_inline)),
                    "fault_source": self.fault_sticks_path.as_posix(),
                }, ensure_ascii=False) + "\n")

                written += 1
                print(f"wrote {written}/{len(inline_ids)} inline={inline_id} has_fault={has_fault}")

        print(f"done: wrote {written} image/mask pairs to {self.data_path}")
        if skipped_empty_labeled:
            print(f"skipped_empty_labeled: {skipped_empty_labeled}")


FaultDetection_DataBuider = FaultDetectionDataBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GN1101 inline fault-detection image/mask dataset.")
    parser.add_argument("--max-inlines", type=int, default=-1, help="Maximum inlines to export. Use -1 for all.")
    parser.add_argument("--positive-only", action="store_true", help="Only export inlines whose mask has fault pixels.")
    parser.add_argument("--keep-empty-labeled", action="store_true", help="Keep labeled inlines whose sticks do not form a mask.")
    parser.add_argument("--mask-radius", type=int, default=3, help="Fault mask dilation radius in pixels.")
    parser.add_argument("--clip", type=float, default=99.0, help="Amplitude clipping percentile.")
    args = parser.parse_args()

    fd = FaultDetectionDataBuilder(mask_radius=args.mask_radius)
    fd.map_data(
        max_inlines=None if args.max_inlines < 0 else args.max_inlines,
        include_negative=not args.positive_only,
        keep_empty_labeled=args.keep_empty_labeled,
        clip_percentile=args.clip,
    )
