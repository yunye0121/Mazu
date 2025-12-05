#!/usr/bin/env python3
"""
Make one loss-vs-time SCATTER plot per variable from one or more CSVs like:

  | <variable> | 1h | 2h | ... | 96h |

Each plot contains multiple scatter series—one per CSV—so you can compare runs.

Usage:
  python make_loss_scatter_plots.py \
    --csv_paths run1.csv run2.csv run3.csv \
    --legend_names "Run 1" "Run 2" "Run 3" \
    --output_dir plots --ext png --dpi 150 --zip
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


def sanitize_filename(s: str) -> str:
    """Turn a variable name into a safe filename chunk."""
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)  # keep letters, numbers, _, -, .
    return s or "var"


def parse_args():
    p = argparse.ArgumentParser(description="Generate one scatter plot per variable from multiple CSVs.")
    p.add_argument(
        "--csv_paths",
        nargs="+",
        required=True,
        help="Path(s) to the CSV file(s). Example: --csv_paths 96hrs_a.csv 96hrs_b.csv",
    )
    p.add_argument(
        "--legend_names",
        nargs="+",
        help="Optional legend_names for each CSV (same length and order as --csv_paths).",
    )
    p.add_argument("--output_dir", default="err_plots", help="Directory to save images.")
    p.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg", "pdf", "svg"], help="Image format.")
    p.add_argument("--dpi", type=int, default=300, help="Image DPI.")
    p.add_argument("--width", type=float, default=6.0, help="Figure width (inches).")
    p.add_argument("--height", type=float, default=6.0, help="Figure height (inches).")
    p.add_argument("--alpha", type=float, default=0.9, help="Marker transparency for scatters.")
    p.add_argument("--markersize", type=float, default=30.0, help="Marker size for scatters (points^2).")
    p.add_argument("--zip", action="store_true", help="Zip all images after saving.")
    return p.parse_args()


def read_csv_with_time_cols(csv_path: Path):
    """Read a CSV and return (df, var_col, time_cols_sorted, hours)."""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path}: expected at least 2 columns (variable + time columns).")

    var_col = df.columns[0]

    # Keep only time columns shaped like '1h', '2h', ... and sort by hour
    time_cols = [c for c in df.columns[1:] if re.fullmatch(r"\d+h", str(c))]
    if not time_cols:
        raise ValueError(f"{csv_path}: no time columns like '1h', '2h', ... were found.")

    time_cols_sorted = sorted(time_cols, key=lambda c: int(str(c).replace("h", "")))
    hours = [int(str(c).replace("h", "")) for c in time_cols_sorted]

    return df, var_col, time_cols_sorted, hours


def collect_all_variables(frames: List[pd.DataFrame], var_cols: List[str]) -> List[str]:
    """Union of variable names across all CSVs (preserving rough order)."""
    seen = set()
    order = []
    for df, var_col in zip(frames, var_cols):
        for v in df[var_col].astype(str).tolist():
            if v not in seen:
                seen.add(v)
                order.append(v)
    return order


def run_label(path: Path) -> str:
    """Return the parent folder name (i.e. the second-to-last component) for a path.

    Example:
    /home/user/experiments/runA/metrics.csv -> "runA"
    """
    # `path.parts` returns a tuple of path components. The last element is the file
    # name; the element before that is the parent folder we want.
    if len(path.parts) >= 3:
        return path.parts[-3]
    # Fallback if for some reason the path has no parent folder.
    return path.stem


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.csv_paths]

    # Determine legend_names for each CSV
    if args.legend_names is not None:
        if len(args.legend_names) != len(csv_paths):
            raise ValueError(
                f"--legend_names must have the same length as --csv_paths "
                f"(got {len(args.legend_names)} legend_names for {len(csv_paths)} CSVs)."
            )
        label_by_file: Dict[Path, str] = {p: lbl for p, lbl in zip(csv_paths, args.legend_names)}
    else:
        # fall back to existing run_label() behavior
        label_by_file: Dict[Path, str] = {p: run_label(p) for p in csv_paths}

    # Read all CSVs and cache their time axes
    frames: List[pd.DataFrame] = []
    var_cols: List[str] = []
    time_cols_by_file: Dict[Path, List[str]] = {}
    hours_by_file: Dict[Path, List[int]] = {}

    for p in csv_paths:
        df, var_col, time_cols_sorted, hours = read_csv_with_time_cols(p)
        frames.append(df)
        var_cols.append(var_col)
        time_cols_by_file[p] = time_cols_sorted
        hours_by_file[p] = hours

    # Build union of all variable names present
    all_vars = collect_all_variables(frames, var_cols)

    # For quick lookup by (file -> variable -> row)
    row_maps: Dict[Path, Dict[str, pd.Series]] = {}
    for p, df, var_col in zip(csv_paths, frames, var_cols):
        # map variable name (as str) to the row (Series)
        row_maps[p] = {
            str(v): row
            for v, row in zip(df[var_col].astype(str).tolist(), df.itertuples(index=False, name=None))
        }

        # NOTE: df.itertuples returns tuples; we'll reconstruct a Series when needed
        df_cols = df.columns.tolist()

        def tuple_to_series(tup):
            return pd.Series(tup, index=df_cols)

        # Replace values with Series
        row_maps[p] = {k: tuple_to_series(v) for k, v in row_maps[p].items()}

    # Generate one plot per variable, adding a scatter for each CSV that contains it
    for var_name in all_vars:
        plt.figure(figsize=(args.width, args.height))

        has_any = False
        for p in csv_paths:
            if var_name not in row_maps[p]:
                continue  # this CSV doesn't have the variable
            row = row_maps[p][var_name]
            time_cols = time_cols_by_file[p]
            hours = hours_by_file[p]

            # pull y values, skip non-numeric safely
            y = pd.to_numeric(row[time_cols], errors="coerce").values
            # mask NaNs so they don't plot
            mask = pd.notna(y)

            if mask.any():
                xs = [h for h, m in zip(hours, mask) if m]
                ys = [val for val, m in zip(y, mask) if m]

                if not xs:
                    continue  # skip empty

                if len(xs) == 1:
                    # Only one data point — draw a large star
                    plt.plot(
                        xs,
                        ys,
                        marker="*",
                        markersize=(args.markersize ** 0.5) * 3.0,
                        linestyle="None",
                        color=None,
                        alpha=args.alpha,
                        label=label_by_file[p],
                        zorder=5,
                    )
                else:
                    # Multiple points — draw a continuous solid line
                    plt.plot(
                        xs,
                        ys,
                        linestyle="-",
                        linewidth=2.0,
                        alpha=args.alpha,
                        label=label_by_file[p],
                    )

                has_any = True

        if not has_any:
            plt.close()
            continue

        plt.xlabel("forecast hour")
        plt.ylabel("loss value")
        plt.tick_params(axis="x", pad=6)  # increase distance between x-ticks and axis
        plt.tick_params(axis="y", pad=6)
        plt.title(f"{var_name}")
        plt.grid(True)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()

        fname = f"{sanitize_filename(var_name)}.{args.ext}"
        plt.savefig(output_dir / fname, dpi=args.dpi)
        plt.close()

    # Optionally zip all images
    if args.zip:
        import shutil

        zip_base = output_dir.parent / f"{output_dir.name}"
        shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)
        print(f"Zipped to: {zip_base}.zip")

    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
