#!/usr/bin/env python
"""
plot_outlier_analysis.py

creates supp figures to investigate outliers and there residuals

One requires GEE to retrieve the chips for Fig 2
Fig 1 are distributions of the residuals for each sensor


Usage
-----
    python plot_outlier_analysis.py              # Fig 1 only
    python plot_outlier_analysis.py --chips      # Fig 1 + Fig 2 (needs GEE)
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler


REPO_ROOT    = Path(__file__).resolve().parent.parent
VAL_DIR      = REPO_ROOT / 'data/GEE_validation_files'
PARAMS_CSV   = REPO_ROOT / 'data/algorithm_params/MASTER_ALGORITHM_PARAMETERS.csv'
OUT_DIR      = REPO_ROOT / 'Figures'

DIST_FIG        = OUT_DIR / 'supp_outlier_distribution.png'
CHIPS_FIG       = OUT_DIR / 'supp_outlier_chips.png'
CHIPS_CLEAN_FIG = OUT_DIR / 'supp_clean_chips.png'

CV_N_SIGMA   = 1          
N_CHIPS      = 12         # chips to show in Fig 2 (2 × 6 grid)
CHIP_PIXELS  = 128
CHIP_RES_M   = 15
RANDOM_SEED  = 42

SENSORS_ORDER = ['L5', 'L7', 'L8', 'L9']

# Quality filters — must match validate_coefficients() in v3
INSITU_TEMP_MIN      = -1.8
INSITU_TEMP_MAX      =  15.0
BT_MIN_KELVIN        =  271.35
BT_MAX_KELVIN        =  288.15
MIN_WIND_SPEED       =    6.0
MIN_PIXEL_COUNT      =    5
BT_STDEV_THRESHOLD   =    1.0
MAX_TIME_DIFF_SECS   = 3600 * 3
EXCLUDED_TYPES       = ['FB', 'CT']

SENSOR_PREFIX = {'L4': 'LT04', 'L5': 'LT05', 'L7': 'LE07', 'L8': 'LC08', 'L9': 'LC09'}


# ─── Data loading ───────────────────────────────────────────────────────────────

def _get_sensor(lid):
    if pd.isna(lid): return None
    s = str(lid)
    if s.startswith('LC09'): return 'L9'
    if s.startswith('LC08'): return 'L8'
    if s.startswith('LE07'): return 'L7'
    if s.startswith('LT05'): return 'L5'
    if s.startswith('LT04'): return 'L4'
    return None


def load_validation_data() -> pd.DataFrame:
    """Load and quality-filter all validation CSVs (replicates validate_coefficients)."""
    dfs = []
    for f in glob.glob(str(VAL_DIR / 'CORA*.csv')):
        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
            if 'TQV' in df.columns and 'TCWV' not in df.columns:
                df = df.rename(columns={'TQV': 'TCWV'})
            for col in ['TEMP', 'brightness_temp', 'ERA5_wind_speed',
                        'brightness_temp_stdev', 'time_diff_seconds']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[(df['TEMP'] >= INSITU_TEMP_MIN) & (df['TEMP'] <= INSITU_TEMP_MAX)]
            df = df[(df['brightness_temp'] >= BT_MIN_KELVIN) &
                    (df['brightness_temp'] <= BT_MAX_KELVIN)]
            if 'ERA5_wind_speed' in df.columns:
                df = df[df['ERA5_wind_speed'] >= MIN_WIND_SPEED]
            df = df[~df['TYPE'].isin(EXCLUDED_TYPES)]
            if 'pixel_count' in df.columns:
                df = df[df['pixel_count'].astype(float) > MIN_PIXEL_COUNT]
            if 'time_diff_seconds' in df.columns:
                df = df[df['time_diff_seconds'].abs() <= MAX_TIME_DIFF_SECS]
            if 'brightness_temp_stdev' in df.columns:
                df = df[df['brightness_temp_stdev'] < BT_STDEV_THRESHOLD]
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        raise RuntimeError(f"No validation data found in {VAL_DIR}")
    combined = pd.concat(dfs, ignore_index=True)
    combined['sensor'] = combined['LANDSAT_ID'].apply(_get_sensor)
    return combined.dropna(subset=['sensor', 'TCWV', 'brightness_temp'])


def load_params() -> pd.DataFrame:
    return pd.read_csv(PARAMS_CSV, index_col='Sensor')


def _reconstruct_scaler(row) -> RobustScaler:
    scaler = RobustScaler()
    scaler.center_ = np.array([row['Scaler_Tb_Center'], row['Scaler_TCWV_Center']])
    scaler.scale_  = np.array([row['Scaler_Tb_Scale'],  row['Scaler_TCWV_Scale']])
    scaler.n_features_in_ = 2
    return scaler


def compute_residuals(val_df: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """
    Apply saved v3 coefficients to derive corrected SST and residuals per sensor.
    Returns val_df extended with: raw_SST_C, corrected_SST_C, residual, is_outlier.
    """
    out = []
    for sensor in SENSORS_ORDER:
        if sensor not in params.index:
            continue
        df_s = val_df[val_df['sensor'] == sensor].copy()
        if df_s.empty:
            continue

        row    = params.loc[sensor]
        scaler = _reconstruct_scaler(row)
        A, B, C = row['Coeff_A_Tb'], row['Coeff_B_TbTCWV'], row['Coeff_Intercept']
        D       = row['D_mean']

        X = scaler.transform(df_s[['brightness_temp', 'TCWV']].values)
        tb, tcwv = X[:, 0], X[:, 1]

        df_s['raw_SST_C']       = A * tb + B * (tb * tcwv) + C - 273.15
        df_s['corrected_SST_C'] = df_s['raw_SST_C'] + D
        df_s['insitu_SST']      = df_s['TEMP']
        df_s['residual']        = df_s['corrected_SST_C'] - df_s['insitu_SST']

        mu, sd = df_s['residual'].mean(), df_s['residual'].std()
        df_s['is_outlier'] = (
            (df_s['residual'] < mu - CV_N_SIGMA * sd) |
            (df_s['residual'] > mu + CV_N_SIGMA * sd)
        )
        df_s['sigma_threshold'] = sd
        out.append(df_s)

    return pd.concat(out, ignore_index=True)


#  Figure 1: residual distributions

def plot_distribution(df: pd.DataFrame, out_path: Path) -> None:

    plt.rcParams.update({
        'font.family':       'Arial',
        'font.size':         9,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    axes = axes.flatten()

    sensor_labels = {'L5': 'Landsat 5', 'L7': 'Landsat 7',
                     'L8': 'Landsat 8', 'L9': 'Landsat 9'}
    panel_letters = ['a', 'b', 'c', 'd']

    for i, sensor in enumerate(SENSORS_ORDER):
        ax  = axes[i]
        dfs = df[df['sensor'] == sensor]
        if dfs.empty:
            ax.axis('off')
            continue

        all_res  = dfs['residual'].values
        inliers  = dfs[~dfs['is_outlier']]['residual'].values
        outliers = dfs[ dfs['is_outlier']]['residual'].values
        sigma    = float(dfs['sigma_threshold'].iloc[0])
        mu_all   = float(np.mean(all_res))
        n_all    = len(all_res)
        n_out    = len(outliers)
        pct_out  = 100.0 * n_out / n_all

        bins  = np.linspace(np.percentile(all_res, 0.5),
                            np.percentile(all_res, 99.5), 45)
        bin_w = bins[1] - bins[0]

        # Grey: full distribution context
        ax.hist(all_res, bins=bins, color='#aaaaaa', alpha=0.45,
                edgecolor='white', linewidth=0.2, zorder=2)
        # Blue: retained inliers
        ax.hist(inliers, bins=bins, color='dodgerblue', alpha=0.80,
                edgecolor='k', linewidth=0.2, label='Retained (within ±1σ)', zorder=3)
        # Red: removed outliers
        ax.hist(outliers, bins=bins, color='red', alpha=0.90,
                edgecolor='k', linewidth=0.2,
                label=f'Removed (±1σ filter, {pct_out:.0f}%)', zorder=4)

        # 1sigmas threshold lines — blended transform keeps x in data coords, y in axes
        blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for xv, lab in [(mu_all - sigma, '−1σ'), (mu_all + sigma, '+1σ')]:
            ax.axvline(xv, color='#d73027', linewidth=1.0, linestyle=':', zorder=6)
            # ax.text(xv, 0.04, lab, transform=blended,
            #         color='#d73027', fontsize=8, ha='center', va='bottom')

        ax.axvline(0, color='k', linewidth=0.6, linestyle='-', alpha=0.3, zorder=2)

        # Stats box at left-middle — most space in a histogram is in the sparse
        # left tail, away from the main peak and the legend
        ax.text(0.03, 0.62,
                f'n = {n_all:,}\n'
                f'σ = {sigma:.2f} °C\n'
                f'kurtosis = {stats.kurtosis(all_res):.2f}\n'
                f'removed: {n_out} ({pct_out:.0f}%)',
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

        ax.set_xlabel('Residual: corrected SST − in-situ (°C)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))

        ax.text(0.04, 0.97, panel_letters[i], transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.7, edgecolor='none'))
        ax.text(0.11, 0.97, sensor_labels[sensor], transform=ax.transAxes,
                fontsize=9, va='top', ha='left', fontweight='bold')

        if i == 0:
            ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#cccccc',
                      loc='upper left', bbox_to_anchor=(0.0, 0.88))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# Sensor config  from get_INSITU-LANDSAT-PAIRS.py 
_SENSORS = {
    'L5': {'SR': 'LANDSAT/LT05/C02/T1_L2', 'TOA': 'LANDSAT/LT05/C02/T1_TOA',
            'TIR': 'B6', 'SR_bands': ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','QA_PIXEL'],
            'TOA_bands': ['B1','B2','B3','B4','B5','B7'],
            'RGB': ['SR_B3','SR_B2','SR_B1'], 'has_pan': False, 'has_cirrus': False},
    'L7': {'SR': 'LANDSAT/LE07/C02/T1_L2', 'TOA': 'LANDSAT/LE07/C02/T1_TOA',
            'TIR': 'B6_VCID_1', 'SR_bands': ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','QA_PIXEL'],
            'TOA_bands': ['B1','B2','B3','B4','B5','B7','B8'],
            'RGB': ['SR_B3','SR_B2','SR_B1'], 'has_pan': True, 'has_cirrus': False},
    'L8': {'SR': 'LANDSAT/LC08/C02/T1_L2', 'TOA': 'LANDSAT/LC08/C02/T1_TOA',
            'TIR': 'B10', 'SR_bands': ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','QA_PIXEL'],
            'TOA_bands': ['B1','B2','B3','B4','B5','B6','B7','B8','B9'],
            'RGB': ['SR_B4','SR_B3','SR_B2'], 'has_pan': True, 'has_cirrus': True},
    'L9': {'SR': 'LANDSAT/LC09/C02/T1_L2', 'TOA': 'LANDSAT/LC09/C02/T1_TOA',
            'TIR': 'B10', 'SR_bands': ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','QA_PIXEL'],
            'TOA_bands': ['B1','B2','B3','B4','B5','B6','B7','B8','B9'],
            'RGB': ['SR_B4','SR_B3','SR_B2'], 'has_pan': True, 'has_cirrus': True},
}


def _build_chip_collection(ee, landsat_id: str, sensor: str):
    """
    Fetch SR+TOA+BT for a single scene and compute final_valid mask using
    the identical logic as mask_water() in get_INSITU-LANDSAT-PAIRS.py.

    """
    cfg   = _SENSORS[sensor]
    parts = landsat_id.split('_')
    path, row = int(parts[2][:3]), int(parts[2][3:])
    date  = f'{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:]}'
    d1    = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    sr_img  = (ee.ImageCollection(cfg['SR'])
                 .filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', landsat_id)).first())
    toa_img = (ee.ImageCollection(cfg['TOA'])
                 .filter(ee.Filter.And(
                     ee.Filter.eq('WRS_PATH', path),
                     ee.Filter.eq('WRS_ROW',  row),
                     ee.Filter.eq('DATE_ACQUIRED', date))).first())

    toa_sel = (toa_img.select(cfg['TOA_bands'])
                      .addBands(toa_img.select([cfg['TIR']], ['brightness_temp'])))
    joined  = sr_img.select(cfg['SR_bands']).addBands(toa_sel)

    # ── Water / cloud / ice mask — identical to mask_water() in the GEE pipeline ──
    qa       = joined.select('QA_PIXEL')
    is_water = qa.bitwiseAnd(1 << 7).neq(0)

    good = (qa.bitwiseAnd(1 << 0).eq(0)                      # NOT fill
              .And(qa.bitwiseAnd(1 << 1).eq(0))              # NOT dilated cloud
              .And(qa.bitwiseAnd(1 << 3).eq(0))              # NOT cloud
              .And(qa.bitwiseAnd(1 << 4).eq(0))              # NOT cloud shadow
              .And(qa.bitwiseAnd(1 << 5).eq(0))              # NOT snow
              .And(qa.rightShift(8).bitwiseAnd(3).lt(2))     # cloud conf < medium
              .And(qa.rightShift(10).bitwiseAnd(3).lt(2))    # shadow conf < medium
              .And(qa.rightShift(12).bitwiseAnd(3).lt(2)))   # snow conf < medium
    if cfg['has_cirrus']:
        good = (good.And(qa.bitwiseAnd(1 << 2).eq(0))
                    .And(qa.rightShift(14).bitwiseAnd(3).lt(2)))

    final_valid = (is_water.And(good)
                           .unmask(0).rename('final_valid'))
    joined = (joined.addBands(final_valid)
                    .set('system:time_start', sr_img.get('system:time_start')))
    return ee.ImageCollection([joined]), date, d1


def _render_chip(da, sensor: str):
    """
    Return (rgb, bt_masked) 

    """
    cfg  = _SENSORS[sensor]
    chip = da.isel(time=0)

    r_raw = chip.sel(band=cfg['RGB'][0]).values.astype(float)
    g_raw = chip.sel(band=cfg['RGB'][1]).values.astype(float)
    b_raw = chip.sel(band=cfg['RGB'][2]).values.astype(float)
    no_data = (r_raw == 0) | np.isnan(r_raw)

    def _to_refl(x):
        return np.where(no_data, np.nan,
                        np.clip(x * 0.0000275 - 0.2, 0.0, 1.0))

    def _stretch(arr):
        """Per-channel percentile stretch """
        valid = arr[np.isfinite(arr)]
        if len(valid) < 10:
            return np.clip(arr / 0.15, 0.0, 1.0)
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        if hi <= lo:
            return np.clip(arr / 0.15, 0.0, 1.0)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    r = _stretch(_to_refl(r_raw))
    g = _stretch(_to_refl(g_raw))
    b = _stretch(_to_refl(b_raw))

    rgb = np.dstack([r, g, b]).astype(np.float32)
    rgb[no_data] = 0.0

    # BT — mask pixels the pipeline would have discarded
    bt = chip.sel(band='brightness_temp').values.astype(float)
    bt[no_data] = np.nan
    band_names = list(da.band.values)
    if 'final_valid' in band_names:
        valid = chip.sel(band='final_valid').values.astype(float)
        bt[valid == 0] = np.nan

    return rgb, bt


# Figure 2: chip thumbnails

def plot_chips(df: pd.DataFrame, out_path: Path,
               min_abs_residual: float = None,
               max_abs_residual: float = None,
               require_inlier: bool = False) -> None:
    """
    N_CHIPS × 2-panel chip grid (RGB / BT) for a stratified random sample.

    """
    try:
        import cubo
        import ee
    except ImportError:
        print('cubo / earthengine-api not available — skipping chip figure.')
        return
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception as e:
        print(f'GEE authentication failed: {e}\nSkipping chip figure.')
        return

    # ── Filter to the requested residual range ───────────────────────────────
    mask = (df['LATITUDE'].notna() &
            df['LONGITUDE'].notna() &
            df['LANDSAT_ID'].notna() &
            df['sensor'].isin(_SENSORS))
    if require_inlier:
        mask = mask & ~df['is_outlier']
    subset = df[mask].copy()
    subset['abs_residual'] = subset['residual'].abs()
    if min_abs_residual is not None:
        subset = subset[subset['abs_residual'] >= min_abs_residual]
    if max_abs_residual is not None:
        subset = subset[subset['abs_residual'] <= max_abs_residual]

    if subset.empty:
        print(f'  No scenes match the residual filter — skipping {out_path.name}.')
        return

    # One row per unique scene; spread evenly across the residual range
    # (ascending for clean scenes so we sample near-zero first, descending for outliers)
    ascending = max_abs_residual is not None and min_abs_residual is None
    by_scene = (subset.sort_values('abs_residual', ascending=ascending)
                      .groupby('LANDSAT_ID', as_index=False).first()
                      .sort_values('abs_residual', ascending=ascending))

    n_avail = len(by_scene)
    n       = min(N_CHIPS, n_avail)
    idx     = np.round(np.linspace(0, n_avail - 1, n)).astype(int)
    samples = by_scene.iloc[idx].reset_index(drop=True)

    CHIPS_PER_ROW = 2
    nrows     = (n + CHIPS_PER_ROW - 1) // CHIPS_PER_ROW
    ncols     = CHIPS_PER_ROW * 2   # alternating RGB / BT columns
    chip_km   = CHIP_PIXELS * CHIP_RES_M / 1000
    buf_px    = 100.0 / CHIP_RES_M

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(CHIPS_PER_ROW * 6.2, nrows * 3.0),
        constrained_layout=True,
        squeeze=False,
    )

    from tqdm import tqdm
    for i, (_, row) in enumerate(tqdm(samples.iterrows(), total=n,
                                       desc='Downloading chips')):
        irow      = i // CHIPS_PER_ROW
        ipair     = i %  CHIPS_PER_ROW
        ax_rgb    = axes[irow, ipair * 2]
        ax_bt     = axes[irow, ipair * 2 + 1]

        lid    = row['LANDSAT_ID']
        lat    = float(row['LATITUDE'])
        lon    = float(row['LONGITUDE'])
        sensor = row['sensor']

        d        = lid.split('_')[3]
        date_str = f'{d[:4]}-{d[4:6]}-{d[6:]}'
        resid    = float(row['residual'])
        title = f'{sensor}  {date_str}    {lat:.1f}°N {lon:.1f}°    residual = {resid:+.2f}°C'

        try:
            coll, d0, d1 = _build_chip_collection(ee, lid, sensor)
            cfg   = _SENSORS[sensor]
            bands = list(dict.fromkeys(
                cfg['RGB'] + ['brightness_temp', 'final_valid']
            ))
            da = cubo.create(
                lat=lat, lon=lon, collection=coll,
                bands=bands, start_date=d0, end_date=d1,
                edge_size=CHIP_PIXELS, resolution=CHIP_RES_M, gee=True,
            )
            rgb, bt = _render_chip(da, sensor)
            h, w    = rgb.shape[:2]
            cx, cy  = w / 2.0, h / 2.0

            # True colour — unmasked
            ax_rgb.imshow(rgb, interpolation='bilinear')
            ax_rgb.set_title(title, fontsize=9, pad=4)

            # BT — masked, RdBu_r (cold=blue, warm=red)
            bt_fin  = bt[np.isfinite(bt)]
            vmin    = np.percentile(bt_fin, 2)  if len(bt_fin) else 265
            vmax    = np.percentile(bt_fin, 98) if len(bt_fin) else 290
            cmap_bt = plt.get_cmap('RdBu_r').copy()
            cmap_bt.set_bad(color='#cccccc')
            im = ax_bt.imshow(bt, cmap=cmap_bt, vmin=vmin, vmax=vmax,
                              interpolation='nearest')
            cb = fig.colorbar(im, ax=ax_bt, shrink=0.95, pad=0.03)
            cb.set_label('Brightness temperature (K)', fontsize=9)
            cb.ax.tick_params(labelsize=8)

            # 100 m extraction buffer circle on both panels
            for ax in [ax_rgb, ax_bt]:
                ax.add_patch(mpatches.Circle(
                    (cx, cy), radius=buf_px,
                    fill=False, edgecolor='yellow', linewidth=1.8, zorder=5,
                ))
                ax.axis('off')

        except Exception as exc:
            for ax in [ax_rgb, ax_bt]:
                ax.set_facecolor('#1a1a1a')
                ax.text(0.5, 0.5, str(exc)[:100], transform=ax.transAxes,
                        ha='center', va='center', fontsize=6, color='tomato',
                        wrap=True)
                ax.axis('off')
            ax_rgb.set_title(title, fontsize=9, pad=4)

    for i in range(n, nrows * CHIPS_PER_ROW):
        irow = i // CHIPS_PER_ROW
        ipair = i % CHIPS_PER_ROW
        axes[irow, ipair * 2    ].set_visible(False)
        axes[irow, ipair * 2 + 1].set_visible(False)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chips', action='store_true',
                    help='Also generate chip thumbnail figure (requires GEE auth + cubo)')
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print('Loading validation data...')
    val_df = load_validation_data()
    print(f'  {len(val_df):,} matchups after quality filters')

    params = load_params()
    print('Applying v3 coefficients and identifying ±1σ outliers...')
    df = compute_residuals(val_df, params)

    for sensor in SENSORS_ORDER:
        sub = df[df['sensor'] == sensor]
        if sub.empty:
            continue
        n_out = sub['is_outlier'].sum()
        pct   = 100 * n_out / len(sub)
        print(f'  {sensor}: {len(sub):5d} total  {n_out:4d} outliers ({pct:.1f}%)')

    print('\nGenerating Figure 1: residual distributions...')
    plot_distribution(df, DIST_FIG)

    if args.chips:
        print('\nGenerating Figure 2: outlier chip thumbnails (|residual| ≥ 2°C)...')
        plot_chips(df, CHIPS_FIG,
                   min_abs_residual=2.0)

        print('\nGenerating Figure 3: clean chip thumbnails (|residual| ≤ 0.5°C)...')
        plot_chips(df, CHIPS_CLEAN_FIG,
                   max_abs_residual=0.5,
                   require_inlier=True)
    else:
        print('\nSkipping chip figures (re-run with --chips to generate, requires GEE).')


if __name__ == '__main__':
    main()
