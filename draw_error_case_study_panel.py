#!/usr/bin/env python3
"""
draw_grid_compare_models_era5.py

Grid figure: ERA5 (first row) + N model rows x V variables columns.
Optionally insert a diff/error row after each model: (Model - ERA5).

Features:
- Per-variable color scale standardized from ERA5 for value maps.
- Each panel has its own colorbar.
- Only left-side row labels + big variable titles on top row.
- Hide all other ticks/axis labels.
- Optional MAE blue box overlay on model rows (--show_mae_box).
- Adjustable spacing between panels (--wspace/--hspace).
- Adjustable spacing for row label and column title (--rowlabel_x/--title_pad).
- Optional super title (--suptitle).
- Optional MAE table output (--mae_out).

First version supports 2D lat/lon variables only.
"""

import os
import json
import csv
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

TIME_DIMS = ["valid_time", "time", "history"]
LAT_NAMES = ["latitude", "lat", "Latitude", "LAT"]
LON_NAMES = ["longitude", "lon", "Longitude", "LON"]


def _first_existing(name_list, ds_or_da):
    for n in name_list:
        if n in getattr(ds_or_da, "dims", {}) or n in getattr(ds_or_da, "coords", {}):
            return n
    return None


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Rename lat/lon to latitude/longitude when needed and sort coords ascending."""
    lat = _first_existing(LAT_NAMES, ds)
    lon = _first_existing(LON_NAMES, ds)

    rename = {}
    if lat and lat != "latitude":
        rename[lat] = "latitude"
    if lon and lon != "longitude":
        rename[lon] = "longitude"
    if rename:
        ds = ds.rename(rename)

    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds


def maybe_select_first_time(da: xr.DataArray) -> xr.DataArray:
    """If any known time-like dim exists, select index 0."""
    for tdim in TIME_DIMS:
        if tdim in da.dims:
            da = da.isel({tdim: 0})
    return da


def select_latlon_range(da: xr.DataArray, lat_range, lon_range, debug_name: str = "") -> xr.DataArray:
    """Crop lat/lon; accept either order from user."""
    if lat_range is None or lon_range is None:
        return da

    lat1, lat2 = lat_range
    lon1, lon2 = lon_range
    lat_min, lat_max = (lat1, lat2) if lat1 <= lat2 else (lat2, lat1)
    lon_min, lon_max = (lon1, lon2) if lon1 <= lon2 else (lon2, lon1)

    out = da.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    if out.sizes.get("latitude", 0) == 0 or out.sizes.get("longitude", 0) == 0:
        try:
            print(
                f"[WARN] Empty crop {debug_name or da.name}: "
                f"lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}. "
                f"Available lat [{float(da.latitude.min())}, {float(da.latitude.max())}], "
                f"lon [{float(da.longitude.min())}, {float(da.longitude.max())}]"
            )
        except Exception:
            print(f"[WARN] Empty crop {debug_name or da.name}.")
    return out


def finite_minmax(arr: xr.DataArray, robust=False, q=(2, 98)):
    """Compute vmin/vmax from a DataArray using finite values."""
    vals = np.asarray(arr.values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    if robust:
        vmin = float(np.percentile(vals, q[0]))
        vmax = float(np.percentile(vals, q[1]))
    else:
        vmin = float(vals.min())
        vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    return vmin, vmax


def symmetric_minmax_from_values(values_1d: np.ndarray, robust=False, q=(2, 98)):
    """Return symmetric (-A, +A) limits from finite 1D values."""
    v = values_1d[np.isfinite(values_1d)]
    if v.size == 0:
        return None, None
    if robust:
        lo = float(np.percentile(v, q[0]))
        hi = float(np.percentile(v, q[1]))
    else:
        lo = float(v.min())
        hi = float(v.max())
    a = max(abs(lo), abs(hi))
    if a == 0:
        a = 1e-6
    return -a, +a


def mae_l1(da_a: xr.DataArray, da_e: xr.DataArray) -> float:
    """Mean absolute error on overlapping finite values."""
    a = np.asarray(da_a.values)
    e = np.asarray(da_e.values)
    mask = np.isfinite(a) & np.isfinite(e)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[mask] - e[mask])))


def normalize_lon_to_match(da_src: xr.DataArray, da_tgt: xr.DataArray) -> xr.DataArray:
    """Shift src longitude range to match target (0..360 vs -180..180)."""
    if "longitude" not in da_src.coords or "longitude" not in da_tgt.coords:
        return da_src

    src_lon = np.asarray(da_src["longitude"].values)
    tgt_lon = np.asarray(da_tgt["longitude"].values)
    if src_lon.size == 0 or tgt_lon.size == 0:
        return da_src

    src_min = float(np.nanmin(src_lon))
    tgt_min = float(np.nanmin(tgt_lon))

    # Target 0..360, src -180..180
    if tgt_min >= 0 and src_min < 0:
        da_src = da_src.assign_coords(longitude=((da_src["longitude"] + 360) % 360))
        da_src = da_src.sortby("longitude")

    # Target -180..180, src 0..360
    if tgt_min < 0 and src_min >= 0:
        da_src = da_src.assign_coords(longitude=(((da_src["longitude"] + 180) % 360) - 180))
        da_src = da_src.sortby("longitude")

    return da_src


def regrid_pred_to_era5(da_p: xr.DataArray, da_e: xr.DataArray) -> xr.DataArray:
    """Interpolate prediction to ERA5 lat/lon grid."""
    da_p = normalize_lon_to_match(da_p, da_e)
    return da_p.interp(
        latitude=da_e["latitude"],
        longitude=da_e["longitude"],
        method="linear",
        kwargs={"fill_value": np.nan},
    )


def open_era5_merged(upper_path: str, sfc_path: str) -> xr.Dataset:
    upper = standardize_latlon(xr.open_dataset(upper_path))
    sfc = standardize_latlon(xr.open_dataset(sfc_path))
    return xr.merge([upper, sfc], compat="no_conflicts", join="outer")


def parse_map_pairs(map_pairs):
    """Parse --map entries like ['surf_2t=t2m', ...] into dict."""
    if not map_pairs:
        return {}
    out = {}
    for p in map_pairs:
        if "=" not in p:
            raise ValueError(f"Bad --map entry '{p}'. Use like --map surf_2t=t2m")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_map_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("--map_json must be a JSON object/dict.")
    return {str(k): str(v) for k, v in obj.items()}


def resolve_era_var(pred_var: str, mapping: dict, map_mode: str) -> str:
    """Return ERA5 var name corresponding to pred_var."""
    if not mapping:
        return pred_var
    if map_mode == "aur_to_era":
        return mapping.get(pred_var, pred_var)
    # era_to_aur: mapping keys are ERA5; values are pred vars
    for era_k, pred_v in mapping.items():
        if pred_v == pred_var:
            return era_k
    return pred_var


def plot_grid(
    rows,               # list of row dicts: {"label": str, "kind": "era"|"model"|"diff", "data": {pred_var: da}, "mae": {pred_var: mae}}
    era_row,            # dict pred_var -> era da
    var_specs,          # list dicts: pred_var, era_var, vmin, vmax, dvmin, dvmax
    outpath_no_ext,
    cmap_value,
    cmap_diff,
    dpi,
    fmt,
    title_fontsize=18,
    rowlabel_fontsize=14,
    cbar_pad=0.18,
    cbar_size="4.5%",
    show_mae_box=False,
    wspace=0.35,
    hspace=0.35,
    suptitle=None,
    suptitle_fontsize=20,
    rowlabel_x=-0.14,
    title_pad=16,
):
    """
    Rows: ERA5 first, then (Model, Diff) pairs if requested.
    Cols: variables.

    Only show:
      - Row labels on left (first column only)
      - Column titles on top row (big)
    Hide all ticks/axis labels.
    Each panel has its own colorbar with adjustable padding.
    """

    nrows = len(rows)
    ncols = len(var_specs)

    fig_w = max(9, 4.7 * ncols)
    fig_h = max(4, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=False)

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    is_vector = fmt.lower() in ["pdf", "svg", "eps"]
    raster_kw = dict(rasterized=is_vector)

    def hide_all(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    def draw_cell(ax, x, y, Z, vmin, vmax, cmap):
        im = ax.pcolormesh(
            x, y, Z,
            shading="nearest",
            vmin=vmin, vmax=vmax, cmap=cmap,
            linewidth=0, edgecolors="none", antialiased=False,
            **raster_kw
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
        fig.colorbar(im, cax=cax)
        return im

    # Column titles (top row only): pred var name
    for j, vs in enumerate(var_specs):
        axes[0, j].set_title(vs["pred_var"], fontsize=title_fontsize, pad=title_pad)

    # Plot all rows
    for i, r in enumerate(rows):
        for j, vs in enumerate(var_specs):
            ax = axes[i, j]
            pred_var = vs["pred_var"]

            da = r["data"].get(pred_var, None)
            if da is None:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            # coordinates always from ERA5 (same grid)
            da_e = era_row[pred_var]
            x = da_e["longitude"].values
            y = da_e["latitude"].values

            if r["kind"] == "diff":
                vmin, vmax = vs["dvmin"], vs["dvmax"]
                if vmin is None:
                    # fallback
                    vmin, vmax = -1.0, 1.0
                draw_cell(ax, x, y, da.values, vmin, vmax, cmap_diff)
            else:
                vmin, vmax = vs["vmin"], vs["vmax"]
                draw_cell(ax, x, y, da.values, vmin, vmax, cmap_value)

            hide_all(ax)

            # Row label only on first column
            if j == 0:
                ax.text(
                    rowlabel_x, 0.5, r["label"],
                    rotation=00, ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=rowlabel_fontsize, fontweight="bold"
                )

            # MAE box: only for model rows (optional)
            if show_mae_box and r["kind"] == "model":
                mae = r.get("mae", {}).get(pred_var, float("nan"))
                mae_str = f"{mae:.4g}" if np.isfinite(mae) else "NaN"
                ax.text(
                    0.02, 0.98, f"MAE={mae_str}",
                    ha="left", va="top",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", alpha=0.75)
                )

    top_margin = 0.88 if suptitle else 0.92
    fig.subplots_adjust(top=top_margin, wspace=wspace, hspace=hspace)

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=0.925)

    os.makedirs(os.path.dirname(outpath_no_ext) or ".", exist_ok=True)
    outpath = f"{outpath_no_ext}.{fmt}"
    raster_fmt = fmt.lower() in ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    plt.savefig(outpath, dpi=dpi if raster_fmt else None, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    p = argparse.ArgumentParser(description="Grid compare: ERA5 (top row) + models (rows) × variables (cols).")
    p.add_argument("--pred_files", type=str, nargs="+", required=True, help="Prediction model netcdf files.")
    p.add_argument("--pred_labels", type=str, nargs="*", default=None, help="Optional labels for pred_files.")
    p.add_argument("--era5_upper_file", type=str, required=True)
    p.add_argument("--era5_sfc_file", type=str, required=True)

    p.add_argument("--output", type=str, default="grid_compare", help="Output path without extension.")
    p.add_argument("--vars", type=str, nargs="+", required=True, help="Variables to plot (pred var names).")

    p.add_argument("--latitude", type=float, nargs=2, metavar=("LAT1", "LAT2"))
    p.add_argument("--longitude", type=float, nargs=2, metavar=("LON1", "LON2"))

    # Colormaps
    p.add_argument("--cmap", type=str, default="cividis", help="Colormap for value maps.")
    p.add_argument("--diff_cmap", type=str, default="RdBu_r", help="Colormap for diff maps (diverging).")

    p.add_argument("--robust", action="store_true", help="Use percentile ranges for scaling (2-98%).")
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--fmt", type=str, default="png")

    # Mapping
    p.add_argument("--map", action="append", default=[], help="Var mapping pair, e.g. --map surf_2t=t2m")
    p.add_argument("--map_json", type=str, default=None, help="Path to JSON mapping dict.")
    p.add_argument("--map_mode", type=str, default="aur_to_era", choices=["aur_to_era", "era_to_aur"])

    # Layout / style
    p.add_argument("--title_fontsize", type=int, default=18, help="Column title fontsize.")
    p.add_argument("--rowlabel_fontsize", type=int, default=14, help="Row label fontsize.")
    p.add_argument("--cbar_pad", type=float, default=0.18, help="Gap between panel and its colorbar.")
    p.add_argument("--cbar_size", type=str, default="4.5%", help="Colorbar width, e.g. '4.5%'.")
    p.add_argument("--show_mae_box", action="store_true", help="Show MAE blue box in model panels.")

    p.add_argument("--wspace", type=float, default=0.35, help="Horizontal space between grid columns.")
    p.add_argument("--hspace", type=float, default=0.35, help="Vertical space between grid rows.")
    p.add_argument("--suptitle", type=str, default=None, help="Super title shown on top of the figure.")
    p.add_argument("--suptitle_fontsize", type=int, default=20, help="Fontsize for super title.")

    p.add_argument("--rowlabel_x", type=float, default=-0.14, help="Row label x-offset (negative = left).")
    p.add_argument("--title_pad", type=float, default=16, help="Padding between column titles and top row panels.")

    # Diff rows
    p.add_argument("--add_diff_rows", action="store_true",
                   help="Insert a diff row after each model row (Model - ERA5).")

    # Optional MAE output table
    p.add_argument("--mae_out", type=str, default=None, help="Write MAE table to CSV/JSON (optional).")

    args = p.parse_args()

    mapping = {}
    if args.map_json:
        mapping.update(load_map_json(args.map_json))
    mapping.update(parse_map_pairs(args.map))

    lat_range = tuple(args.latitude) if args.latitude is not None else None
    lon_range = tuple(args.longitude) if args.longitude is not None else None

    era = open_era5_merged(args.era5_upper_file, args.era5_sfc_file)

    labels = args.pred_labels
    if not labels or len(labels) != len(args.pred_files):
        labels = [f"Model{i+1}" for i in range(len(args.pred_files))]

    # Prepare ERA5 slices + per-column scales for value maps
    era_row = {}
    var_specs = []
    for pred_var in args.vars:
        era_var = resolve_era_var(pred_var, mapping, args.map_mode)
        if era_var not in era.data_vars:
            print(f"[WARN] ERA5 missing '{era_var}' (for pred '{pred_var}'), skip column.")
            continue

        da_e = maybe_select_first_time(era[era_var])
        da_e = select_latlon_range(da_e, lat_range, lon_range, debug_name=f"ERA5:{era_var}")

        if set(da_e.dims) != {"latitude", "longitude"}:
            print(f"[WARN] ERA5 '{era_var}' dims {da_e.dims} not 2D lat/lon. Skipping (first version).")
            continue

        vmin, vmax = finite_minmax(da_e, robust=args.robust)
        if vmin is None:
            print(f"[WARN] ERA5 '{era_var}' all-NaN. Skipping.")
            continue

        era_row[pred_var] = da_e
        var_specs.append(dict(pred_var=pred_var, era_var=era_var, vmin=vmin, vmax=vmax, dvmin=None, dvmax=None))

    if not var_specs:
        raise SystemExit("[ERROR] No valid variables to plot after filtering.")

    # Build model rows, compute MAE and diffs
    models = []
    mae_records = []
    # Collect diffs per variable to set symmetric diff scales
    diffs_collect = {vs["pred_var"]: [] for vs in var_specs}

    for path, lab in zip(args.pred_files, labels):
        ds = standardize_latlon(xr.open_dataset(path))
        mdata = {}
        mmae = {}
        mdiff = {}

        for vs in var_specs:
            pred_var = vs["pred_var"]
            era_var = vs["era_var"]

            if pred_var not in ds.data_vars:
                print(f"[WARN] {lab}: missing '{pred_var}' in {os.path.basename(path)}")
                continue

            da_p = maybe_select_first_time(ds[pred_var])
            da_p = select_latlon_range(da_p, lat_range, lon_range, debug_name=f"{lab}:{pred_var}")

            if set(da_p.dims) != {"latitude", "longitude"}:
                print(f"[WARN] {lab}:{pred_var} dims {da_p.dims} not 2D lat/lon. Skipping (first version).")
                continue

            da_e = era_row[pred_var]
            da_p_rg = regrid_pred_to_era5(da_p, da_e)

            mae = mae_l1(da_p_rg, da_e)
            diff = (da_p_rg - da_e)

            mdata[pred_var] = da_p_rg
            mmae[pred_var] = mae
            mdiff[pred_var] = diff

            # collect diff values for scaling
            diffs_collect[pred_var].append(np.asarray(diff.values).ravel())

            mae_records.append({
                "row_label": lab,
                "pred_file": os.path.basename(path),
                "pred_var": pred_var,
                "era_var": era_var,
                "mae": mae,
            })

        models.append({"label": lab, "data": mdata, "mae": mmae, "diff": mdiff})

    # Determine diff color scales per variable (symmetric around 0)
    for vs in var_specs:
        pv = vs["pred_var"]
        if diffs_collect.get(pv) and len(diffs_collect[pv]) > 0:
            allv = np.concatenate(diffs_collect[pv], axis=0)
            dvmin, dvmax = symmetric_minmax_from_values(allv, robust=args.robust)
            vs["dvmin"], vs["dvmax"] = dvmin, dvmax
        else:
            vs["dvmin"], vs["dvmax"] = None, None

    # Build row list for plotting
    rows = []
    # ERA5 row
    rows.append({"label": "ERA5", "kind": "era", "data": {pv: era_row[pv] for pv in era_row}, "mae": {}})

    # Model rows (and optional diff rows)
    for m in models:
        rows.append({"label": m["label"], "kind": "model", "data": m["data"], "mae": m["mae"]})
        if args.add_diff_rows:
            rows.append({"label": f"Diff\n{m['label']}-ERA5", "kind": "diff", "data": m["diff"], "mae": {}})

    # Plot
    plot_grid(
        rows=rows,
        era_row=era_row,
        var_specs=var_specs,
        outpath_no_ext=args.output,
        cmap_value=args.cmap,
        cmap_diff=args.diff_cmap,
        dpi=args.dpi,
        fmt=args.fmt,
        title_fontsize=args.title_fontsize,
        rowlabel_fontsize=args.rowlabel_fontsize,
        cbar_pad=args.cbar_pad,
        cbar_size=args.cbar_size,
        show_mae_box=args.show_mae_box,
        wspace=args.wspace,
        hspace=args.hspace,
        suptitle=args.suptitle,
        suptitle_fontsize=args.suptitle_fontsize,
        rowlabel_x=args.rowlabel_x,
        title_pad=args.title_pad,
    )

    # Optional MAE table
    if args.mae_out:
        os.makedirs(os.path.dirname(args.mae_out) or ".", exist_ok=True)
        if args.mae_out.lower().endswith(".json"):
            with open(args.mae_out, "w", encoding="utf-8") as f:
                json.dump(mae_records, f, indent=2)
        else:
            keys = ["row_label", "pred_file", "pred_var", "era_var", "mae"]
            with open(args.mae_out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in mae_records:
                    w.writerow(r)
        print(f"Saved MAE table: {args.mae_out}")


if __name__ == "__main__":
    main()
