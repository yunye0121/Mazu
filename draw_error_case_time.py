#!/usr/bin/env python3
"""
draw_time_series_split.py

Generates time-series grid plots where EACH TIME STEP comes from DIFFERENT files.

Features:
  - Separate ERA5/Model files per timestamp (via JSON manifest)
  - Adjust spacing (--wspace, --hspace)
  - Consolidated colorbars (--single_cbar)
  - Global vmin/vmax scaling across all time steps
  - Auto-centering of title over the plot area

Usage:
  python3 draw_time_series_split.py --manifest list.json --vars surf_2t ...
"""

import os
import json
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Helpers ---
TIME_DIMS = ["valid_time", "time", "history"]
LAT_NAMES = ["latitude", "lat", "Latitude", "LAT"]
LON_NAMES = ["longitude", "lon", "Longitude", "LON"]
LEVEL_DIMS = ["pressure_level", "level", "plev", "isobaricInhPa"]

def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Ensure dimensions are named latitude/longitude and sorted."""
    for n in LAT_NAMES:
        if n in ds.coords or n in ds.dims:
            ds = ds.rename({n: "latitude"})
            break
    for n in LON_NAMES:
        if n in ds.coords or n in ds.dims:
            ds = ds.rename({n: "longitude"})
            break
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds

def open_era5_merged(upper_path, sfc_path):
    """Open upper and surface ERA5 files and merge them."""
    u = standardize_latlon(xr.open_dataset(upper_path))
    s = standardize_latlon(xr.open_dataset(sfc_path))
    return xr.merge([u, s], compat="no_conflicts", join="outer")

def select_latlon_range(da, lat_range, lon_range):
    """Crop DataArray to specific lat/lon box."""
    if not lat_range or not lon_range:
        return da
    lat1, lat2 = lat_range
    lon1, lon2 = lon_range
    lat_min, lat_max = sorted([lat1, lat2])
    lon_min, lon_max = sorted([lon1, lon2])
    return da.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

def parse_var_and_level(varname):
    """Parse 'atmos_z_500' -> ('atmos_z', 500.0)."""
    parts = varname.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), float(parts[-1])
    return varname, None

def select_level(da, lev):
    """Select pressure level if requested."""
    if lev is None:
        return da
    for d in LEVEL_DIMS:
        if d in da.dims:
            try:
                return da.sel({d: lev})
            except:
                try:
                    return da.sel({d: lev}, method="nearest")
                except:
                    pass
    return da

def get_first_time(da):
    """Select the first time step found in the array."""
    for t in TIME_DIMS:
        if t in da.dims:
            return da.isel({t: 0})
    return da

def regrid_to_era(da_p, da_e):
    """Regrid prediction to match ERA5 grid."""
    # Handle 0..360 vs -180..180 longitude mismatch
    if "longitude" in da_p.coords:
        if da_e.longitude.min() < 0 and da_p.longitude.min() >= 0:
            da_p = da_p.assign_coords(longitude=(((da_p.longitude + 180) % 360) - 180)).sortby("longitude")
    
    return da_p.interp(
        latitude=da_e.latitude, 
        longitude=da_e.longitude, 
        method="linear", 
        kwargs={"fill_value": np.nan}
    )

def finite_minmax(vals, robust=False):
    """Calculate min/max ignoring NaNs."""
    v = vals[np.isfinite(vals)]
    if v.size == 0: return 0, 1
    if robust:
        return np.percentile(v, 2), np.percentile(v, 98)
    return v.min(), v.max()

def symmetric_minmax(vals, robust=False):
    """Calculate symmetric min/max centered on 0."""
    v = vals[np.isfinite(vals)]
    if v.size == 0: return -1, 1
    mx = np.percentile(np.abs(v), 98) if robust else np.abs(v).max()
    if mx == 0: mx = 1e-6
    return -mx, mx

# --- Plotting ---

def plot_grid(
    col_data, 
    var_name, 
    outpath, 
    vmin, vmax, 
    dvmin, dvmax, 
    cmap="viridis", 
    diff_cmap="RdBu_r",
    wspace=0.05, hspace=0.1,
    single_cbar=False
):
    cols = len(col_data)
    # Figure size: Scale width by number of columns
    fig_w = max(10, 4.0 * cols)
    fig_h = 9
    
    fig, axes = plt.subplots(3, cols, figsize=(fig_w, fig_h))
    
    # Ensure axes is always 2D array [row, col]
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    labels = ["ERA5 (Truth)", "Prediction", "Diff (Pred-ERA)"]
    
    # Store mappables for consolidated colorbar
    im_val = None
    im_diff = None

    for j, item in enumerate(col_data):
        ax_col = axes[:, j]
        
        # Column Header (Time)
        ax_col[0].set_title(f"+{item['hour']}h", fontsize=14, pad=16, fontweight='bold')

        # Rows: 0=ERA, 1=Pred, 2=Diff
        rows_to_plot = [
            ("era",  item['era'],  vmin, vmax, cmap),
            ("pred", item['pred'], vmin, vmax, cmap),
            ("diff", item['diff'], dvmin, dvmax, diff_cmap)
        ]

        for i, (kind, da, vm, vx, cm) in enumerate(rows_to_plot):
            ax = ax_col[i]
            
            if da is None:
                ax.text(0.5, 0.5, "MISSING", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            im = ax.pcolormesh(
                da.longitude, da.latitude, da.values, 
                vmin=vm, vmax=vx, cmap=cm, shading='nearest'
            )
            
            # Save reference for external colorbars
            if i < 2: im_val = im   # ERA or Pred use value map
            else:     im_diff = im  # Diff uses diff map

            # Clean up ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Row Labels (First column only)
            if j == 0:
                ax.text(-0.15, 0.5, labels[i], 
                        rotation=90, va='center', ha='center',
                        transform=ax.transAxes, 
                        fontweight='bold', fontsize=14)

            # Per-panel colorbars (if NOT using single_cbar)
            if not single_cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

    # Adjust layout
    # If single_cbar is True, we leave more room on the right
    right_margin = 0.90 if single_cbar else 0.98
    plt.subplots_adjust(wspace=wspace, hspace=hspace, right=right_margin, top=0.92)

    # --- Consolidated Colorbars ---
    if single_cbar:
        # 1. Main Value Colorbar (Spans ERA & Pred rows)
        if im_val:
            # Position: [left, bottom, width, height]
            # Coordinates are figure-relative (0 to 1)
            cax_val = fig.add_axes([0.92, 0.40, 0.015, 0.45]) 
            cb_val = fig.colorbar(im_val, cax=cax_val)
            cb_val.set_label("Value", fontsize=12)

        # 2. Diff Colorbar (Spans Diff row)
        if im_diff:
            cax_diff = fig.add_axes([0.92, 0.12, 0.015, 0.22]) 
            cb_diff = fig.colorbar(im_diff, cax=cax_diff)
            cb_diff.set_label("Difference", fontsize=12)

    # --- Title Centering Logic ---
    # We calculate the visual center of the plot area (ignoring the sidebar)
    if single_cbar and cols > 0:
        pos_left = axes[0, 0].get_position()
        pos_right = axes[0, -1].get_position()
        visual_center_x = (pos_left.x0 + pos_right.x1) / 2
        fig.suptitle(f"Variable: {var_name}", x=visual_center_x, y=1.02, fontsize=18)
    else:
        fig.suptitle(f"Variable: {var_name}", y=1.02, fontsize=18)
    
    # Save
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {outpath}")


# --- Main ---
def main():
    p = argparse.ArgumentParser(description="Plot time-series grid from split files.")
    p.add_argument("--manifest", type=str, required=True, help="JSON file mapping hours to file paths.")
    p.add_argument("--vars", type=str, nargs="+", required=True, help="Variables to plot (e.g. surf_2t atmos_z_500).")
    p.add_argument("--output_prefix", type=str, default="split_out", help="Output filename prefix.")
    
    # Layout Options
    p.add_argument("--wspace", type=float, default=0.05, help="Horizontal space between subplots.")
    p.add_argument("--hspace", type=float, default=0.1, help="Vertical space between subplots.")
    p.add_argument("--single_cbar", action="store_true", help="Use a single colorbar on the right instead of one per panel.")

    # Data Selection
    p.add_argument("--latitude", type=float, nargs=2, help="Lat range (e.g. 20 50)")
    p.add_argument("--longitude", type=float, nargs=2, help="Lon range (e.g. 100 140)")
    p.add_argument("--map", action="append", default=[], help="Map vars: --map surf_2t=2t")
    p.add_argument("--robust", action="store_true", help="Use 2nd/98th percentile for scaling.")
    
    args = p.parse_args()

    # 1. Load Manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    # Sort manifest by hour to ensure chronological order in plot
    manifest.sort(key=lambda x: x.get('hour', 0))

    # Parse variable mapping
    mapping = {}
    for m in args.map:
        if "=" in m:
            k, v = m.split("=", 1)
            mapping[k.strip()] = v.strip()

    lat_r = args.latitude
    lon_r = args.longitude

    # 2. Process Variables
    for var_token in args.vars:
        print(f"--- Processing {var_token} ---")
        base_var, lev = parse_var_and_level(var_token)
        era_var = mapping.get(base_var, base_var)

        col_data = []
        all_vals = []   # To calculate global vmin/vmax
        all_diffs = []  # To calculate global diff scale

        # Loop through timestamps in manifest
        for entry in manifest:
            hour = entry.get('hour', '?')
            
            # Open files safely
            try:
                ds_p = standardize_latlon(xr.open_dataset(entry['pred']))
                ds_e = open_era5_merged(entry['era_upper'], entry['era_sfc'])
            except Exception as e:
                print(f"[WARN] Failed to open files for T+{hour}h: {e}")
                col_data.append({'hour': hour, 'era': None, 'pred': None, 'diff': None})
                continue

            # Check variables
            if base_var not in ds_p:
                print(f"[WARN] '{base_var}' missing in Pred file for T+{hour}h")
                col_data.append({'hour': hour, 'era': None, 'pred': None, 'diff': None})
                continue
            if era_var not in ds_e:
                print(f"[WARN] '{era_var}' missing in ERA5 file for T+{hour}h")
                col_data.append({'hour': hour, 'era': None, 'pred': None, 'diff': None})
                continue
            
            # Select Data
            da_p = get_first_time(ds_p[base_var])
            da_e = get_first_time(ds_e[era_var])

            da_p = select_level(select_latlon_range(da_p, lat_r, lon_r), lev)
            da_e = select_level(select_latlon_range(da_e, lat_r, lon_r), lev)

            if da_p.size == 0 or da_e.size == 0:
                print(f"[WARN] Empty data after crop/level select for T+{hour}h")
                col_data.append({'hour': hour, 'era': None, 'pred': None, 'diff': None})
                continue

            # Regrid & Diff
            da_p_reg = regrid_to_era(da_p, da_e)
            diff = da_p_reg - da_e

            col_data.append({
                'hour': hour,
                'era': da_e,
                'pred': da_p_reg,
                'diff': diff
            })

            all_vals.append(da_e.values.ravel())
            all_vals.append(da_p_reg.values.ravel())
            all_diffs.append(diff.values.ravel())

        # 3. Global Scaling & Plot
        if not all_vals:
            print(f"No valid data found for {var_token}. Skipping.")
            continue
            
        big_v = np.concatenate(all_vals)
        big_d = np.concatenate(all_diffs)
        
        vmin, vmax = finite_minmax(big_v, robust=args.robust)
        dvmin, dvmax = symmetric_minmax(big_d, robust=args.robust)

        # Smart Colormap Selection
        if "surf" in base_var:
            cmap = "inferno" # or 'magma', 'viridis'
        elif "atmos" in base_var:
            cmap = "plasma"
        else:
            cmap = "viridis"

        out_filename = f"{args.output_prefix}_{var_token}.pdf"
        
        plot_grid(
            col_data, 
            var_name=var_token, 
            outpath=out_filename, 
            vmin=vmin, vmax=vmax, 
            dvmin=dvmin, dvmax=dvmax, 
            cmap=cmap,
            wspace=args.wspace,
            hspace=args.hspace,
            single_cbar=args.single_cbar
        )

if __name__ == "__main__":
    main()