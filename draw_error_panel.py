#!/usr/bin/env python3
"""
Generate loss-vs-time plots from one or more CSVs shaped like:

  | <variable> | 1h | 2h | ... | 96h |

Features:
- One figure per variable (default), OR:
- Combine variables into one axes (--combine), OR:
- Panel grid with one subplot per variable (--panel)
- Single shared legend at the top (--legend_top)
- Optional custom legend labels (--legend_names ...)
- Select subset of variables (--vars ...)

Usage examples:

# Panel of 3 variables, shared legend at top, custom labels
python make_loss_scatter_plots.py \
  --csv_paths runs/a/metrics.csv runs/b/metrics.csv runs/c/metrics.csv \
  --legend_names "Baseline" "Improved v2" "Experimental" \
  --vars T2M RH W10 \
  --panel --panel_cols 3 --legend_top --legend_cols 3 \
  --output_dir plots

# Default legend, one figure per variable (filtered)
python make_loss_scatter_plots.py \
  --csv_paths runA.csv runB.csv \
  --vars T2M RH W10 \
  --output_dir plots
"""

import argparse
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------- Utilities ---------------------------

def sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)  # keep letters, numbers, _, -, .
    return s or "var"


def parse_args():
    p = argparse.ArgumentParser(description="Generate loss-vs-time plots from multiple CSVs (per-var, combined, or panel).")
    p.add_argument("--csv_paths", nargs="+", required=True,
                   help="Path(s) to the CSV file(s). Example: --csv_paths 96hrs_a.csv 96hrs_b.csv")
    p.add_argument("--output_dir", default="err_plots", help="Directory to save images.")
    p.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg", "pdf", "svg"], help="Image format.")
    p.add_argument("--dpi", type=int, default=300, help="Image DPI.")
    p.add_argument("--width", type=float, default=16.0, help="Figure width (inches). For panel: width of WHOLE figure.")
    p.add_argument("--height", type=float, default=8.0, help="Figure height (inches). For panel: height of WHOLE figure.")
    p.add_argument("--alpha", type=float, default=0.9, help="Marker/line transparency.")
    p.add_argument("--markersize", type=float, default=30.0, help="Marker size for single-point series (points^2).")
    p.add_argument("--zip", action="store_true", help="Zip all images after saving.")

    # selection & layouts
    p.add_argument("--vars", nargs="+", default=None,
                   help="Only include these variables (by exact name in the CSV first column).")
    p.add_argument("--combine", action="store_true",
                   help="Put all selected variables together in ONE axes (legacy combine).")
    p.add_argument("--legend_top", action="store_true",
                   help="Place a single, shared legend at the top (deduplicated by run label).")
    p.add_argument("--legend_cols", type=int, default=3,
                   help="Number of columns for the top legend.")

    # NEW: panel (grid of subplots)
    p.add_argument("--panel", action="store_true",
                   help="Create a panel grid: one subplot per selected variable.")
    p.add_argument("--panel_cols", type=int, default=3,
                   help="Columns in the panel grid (rows auto).")
    p.add_argument("--panel_title", default=None,
                   help="Optional overall title for the panel figure.")
    p.add_argument("--sharey", action="store_true",
                   help="Share Y axis across panel subplots. (Default off = independent axes)")

    # NEW: custom legend labels
    p.add_argument("--legend_names", nargs="+", default=None,
                   help="Optional custom legend labels, one per CSV path. "
                        "If omitted or shorter than CSV count, missing labels fall back to folder/filename.")

    return p.parse_args()


def read_csv_with_time_cols(csv_path: Path) -> Tuple[pd.DataFrame, str, List[str], List[int]]:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path}: expected at least 2 columns (variable + time columns).")

    var_col = df.columns[0]
    time_cols = [c for c in df.columns[1:] if re.fullmatch(r"\d+h", str(c))]
    if not time_cols:
        raise ValueError(f"{csv_path}: no time columns like '1h', '2h', ... were found.")

    time_cols_sorted = sorted(time_cols, key=lambda c: int(str(c).replace("h", "")))
    hours = [int(str(c).replace("h", "")) for c in time_cols_sorted]
    return df, var_col, time_cols_sorted, hours


def collect_all_variables(frames: List[pd.DataFrame], var_cols: List[str]) -> List[str]:
    seen, order = set(), []
    for df, var_col in zip(frames, var_cols):
        for v in df[var_col].astype(str).tolist():
            if v not in seen:
                seen.add(v)
                order.append(v)
    return order


def default_run_label(path: Path) -> str:
    """Fallback legend label from path."""
    return path.parts[-2] if len(path.parts) >= 2 else path.stem


def label_for_index(i: int, path: Path, custom_names: Optional[List[str]]) -> str:
    """Return legend label for CSV index i, using custom name if given, else fallback."""
    if custom_names and i < len(custom_names):
        return str(custom_names[i])
    return default_run_label(path)


def plot_one_variable(ax,
                      var_name: str,
                      csv_paths: List[Path],
                      row_maps: Dict[Path, Dict[str, pd.Series]],
                      time_cols_by_file: Dict[Path, List[str]],
                      hours_by_file: Dict[Path, List[int]],
                      alpha: float,
                      markersize: float,
                      legend_names: Optional[List[str]]) -> bool:
    """Add the series for this variable (across all CSVs) onto ax. Returns True if anything was plotted."""
    has_any = False
    for i, p in enumerate(csv_paths):
        if var_name not in row_maps[p]:
            continue
        row = row_maps[p][var_name]
        time_cols = time_cols_by_file[p]
        hours = hours_by_file[p]

        y = pd.to_numeric(row[time_cols], errors="coerce").values
        mask = pd.notna(y)
        if not mask.any():
            continue

        xs = [h for h, m in zip(hours, mask) if m]
        ys = [val for val, m in zip(y, mask) if m]
        if not xs:
            continue

        label = label_for_index(i, p, legend_names)

        if len(xs) == 1:
            # Single point: plot a large star
            ax.plot(xs, ys,
                    marker='*',
                    markersize=(markersize ** 0.5) * 3.0,
                    linestyle='None',
                    alpha=alpha,
                    label=label,
                    zorder=5)
        else:
            # Multiple points: line
            ax.plot(xs, ys,
                    linestyle='-',
                    linewidth=2.0,
                    alpha=alpha,
                    label=label)
        has_any = True
    return has_any


def dedup_legend_from_axes(axes: List[plt.Axes]):
    """Return (handles, labels) with duplicate labels removed (keep first) across multiple axes."""
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    seen = {}
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen[lab] = h
    new_labels = list(seen.keys())
    new_handles = [seen[k] for k in new_labels]
    return new_handles, new_labels


# --------------------------- Main ---------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.csv_paths]

    # Validate legend_names length (warn, but allow fallback)
    if args.legend_names and len(args.legend_names) < len(csv_paths):
        print(f"Warning: --legend_names has {len(args.legend_names)} entries but there are {len(csv_paths)} CSVs. "
              f"Missing labels will use folder/filename.")

    # Read CSVs and cache their time axes
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

    # Build union of all variable names present, then filter if --vars provided
    all_vars = collect_all_variables(frames, var_cols)
    if args.vars:
        selected = set(map(str, args.vars))
        plot_vars = [v for v in all_vars if v in selected]
        missing = sorted(list(selected - set(plot_vars)))
        if missing:
            print(f"Warning: variables not found in any CSV: {', '.join(missing)}")
    else:
        plot_vars = all_vars

    # For quick lookup by (file -> variable -> Series)
    row_maps: Dict[Path, Dict[str, pd.Series]] = {}
    for p, df, var_col in zip(csv_paths, frames, var_cols):
        cols = df.columns.tolist()
        to_series = lambda tup: pd.Series(tup, index=cols)
        row_maps[p] = {
            str(v): to_series(t)
            for v, t in zip(df[var_col].astype(str).tolist(),
                            df.itertuples(index=False, name=None))
        }

    # ================= PANEL MODE =================
    if args.panel and plot_vars:
        n = len(plot_vars)
        ncols = max(1, args.panel_cols)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(args.width, args.height),
                                 sharey=args.sharey, squeeze=False)

        any_plotted = False
        flat_axes = [ax for row in axes for ax in row]

        for i, var_name in enumerate(plot_vars):
            ax = flat_axes[i]
            did = plot_one_variable(ax, var_name, csv_paths, row_maps, time_cols_by_file, hours_by_file,
                                    args.alpha, args.markersize, args.legend_names)
            if did:
                any_plotted = True
                ax.set_title(str(var_name))
                ax.set_xlabel("forecast hour")
                ax.set_ylabel("loss value")
                ax.grid(True)
            else:
                ax.set_visible(False)

        # Hide extra axes, if any
        for j in range(len(plot_vars), len(flat_axes)):
            flat_axes[j].set_visible(False)

        if any_plotted:
            if args.legend_top:
                h, l = dedup_legend_from_axes([ax for ax in flat_axes if ax.get_visible()])
                fig.legend(handles=h, labels=l, loc="upper center",
                           ncol=args.legend_cols, frameon=True, bbox_to_anchor=(0.5, 1))
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                # put legend inside the last visible subplot
                for ax in reversed(flat_axes):
                    if ax.get_visible():
                        ax.legend(loc="best", frameon=True)
                        break
                plt.tight_layout()

            if args.panel_title:
                fig.suptitle(args.panel_title, y=0.995)
                plt.tight_layout(rect=[0, 0, 1, 0.96])

            fname = f"panel-[{'-'.join(sanitize_filename(v) for v in plot_vars)}].{args.ext}"
            fig.savefig(output_dir / fname, dpi=args.dpi)
            plt.close(fig)

        # Optionally zip all images
        if args.zip:
            import shutil
            zip_base = output_dir.parent / f"{output_dir.name}"
            shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)
            print(f"Zipped to: {zip_base}.zip")

        print(f"Saved plots to: {output_dir.resolve()}")
        return

    # ================= COMBINE (single axes) OR PER-VAR =================
    if args.combine and plot_vars:
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        any_plotted = False
        for var_name in plot_vars:
            did = plot_one_variable(ax, var_name, csv_paths, row_maps, time_cols_by_file, hours_by_file,
                                    args.alpha, args.markersize, args.legend_names)
            any_plotted = any_plotted or did

        if any_plotted:
            ax.set_xlabel("forecast hour")
            ax.set_ylabel("loss value")
            ax.set_title(" / ".join(plot_vars) if len(plot_vars) <= 3 else f"{len(plot_vars)} variables")
            ax.grid(True)

            if args.legend_top:
                h, l = ax.get_legend_handles_labels()
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                fig.legend(handles=h, labels=l, loc="upper center",
                           ncol=args.legend_cols, frameon=True, bbox_to_anchor=(0.5, 1.02))
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                ax.legend(loc="best", frameon=True)
                plt.tight_layout()

            fname = f"combine_[{'-'.join(sanitize_filename(v) for v in plot_vars)}].{args.ext}"
            fig.savefig(output_dir / fname, dpi=args.dpi)
            plt.close(fig)

    else:
        # One figure per variable (filtered by --vars if provided)
        for var_name in plot_vars:
            fig, ax = plt.subplots(figsize=(args.width, args.height))

            if not plot_one_variable(ax, var_name, csv_paths, row_maps, time_cols_by_file, hours_by_file,
                                     args.alpha, args.markersize, args.legend_names):
                plt.close(fig)
                continue

            ax.set_xlabel("forecast hour")
            ax.set_ylabel("loss value")
            ax.set_title(f"{var_name}")
            ax.grid(True)

            if args.legend_top:
                h, l = ax.get_legend_handles_labels()
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                fig.legend(handles=h, labels=l, loc="upper center",
                           ncol=args.legend_cols, frameon=True, bbox_to_anchor=(0.5, 1.02))
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                ax.legend(loc="best", frameon=True)
                plt.tight_layout()

            fname = f"{sanitize_filename(var_name)}.{args.ext}"
            fig.savefig(output_dir / fname, dpi=args.dpi)
            plt.close(fig)

    # Optionally zip all images
    if args.zip:
        import shutil
        zip_base = output_dir.parent / f"{output_dir.name}"
        shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)
        print(f"Zipped to: {zip_base}.zip")

    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
