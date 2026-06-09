#!/usr/bin/env python
"""
Supplementary figure: MERRA-2 atmospheric profile sampling coverage and
input data distributions for libRadtran coefficient derivation (1985–2025).

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats as scipy_stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_CSV   = REPO_ROOT / "data/MERRA_2_sampling/MERRA-2_sampling_data_1985-2025.csv"
OUTPUT_FIG = REPO_ROOT / "Figures/MERRA2_sampling_fast.png"


plt.rcParams.update({
    'font.family':       'Arial',
    'font.size':         10,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
})

_cmap_m = plt.get_cmap('YlGnBu_r')
MONTH_COLORS = {m: _cmap_m(v)
                for m, v in zip([5, 6, 7, 8, 9], np.linspace(0.85, 0.18, 5))}
MONTH_NAMES  = {5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep'}
MONTHS_ORDER = [5, 6, 7, 8, 9]

COLOR_VLINE  = '#d73027'
SST_THRESH_C = 272.0 - 273.15


df = pd.read_csv(DATA_CSV, parse_dates=['timestamp'])
df['month'] = df['timestamp'].dt.month
df['year']  = df['timestamp'].dt.year

locs = df[['longitude', 'latitude']].dropna()
vals = np.vstack([locs['longitude'].values, locs['latitude'].values])
density = scipy_stats.gaussian_kde(vals)(vals)
df_map = locs.copy()
df_map['density'] = density
df_map = df_map.sort_values('density')

# 3-row × 2-col GridSpec

fig = plt.figure(figsize=(7.0, 3.6), dpi=96)
gs  = gridspec.GridSpec(
    3, 2,
    figure=fig,
    left=0.08, right=0.97,
    top=0.88,  bottom=0.11,
    hspace=0.52, wspace=0.1,
    width_ratios=[1.6, 1],
    height_ratios=[1, 1, 1],
)

# 
if HAS_CARTOPY:
    proj   = ccrs.NorthPolarStereo(central_longitude=-40)
    ax_map = fig.add_subplot(gs[:, 0], projection=proj)
    ax_map.set_extent([-61, -19, 59, 81], crs=ccrs.PlateCarree())

    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5, zorder=0)
    ax_map.add_feature(cfeature.LAND,  facecolor='white', edgecolor='k',
                       linewidth=0.5, zorder=1)
    ax_map.coastlines(linewidth=0.7, color='k', zorder=2)

    gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                          alpha=0.5, linestyle='--', zorder=3,
                          x_inline=False, y_inline=False)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.FixedLocator([-60, -55, -45, -35, -25])
    gl.ylocator     = mticker.FixedLocator([62, 68, 74, 80])
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    sc = ax_map.scatter(
        df_map['longitude'], df_map['latitude'],
        c=df_map['density'], cmap='YlGnBu_r',
        s=7, alpha=0.75, edgecolors='k', linewidths=0.3,
        transform=ccrs.PlateCarree(), zorder=4,
    )

    # Thin horizontal colorbar
    mp  = ax_map.get_position()
    cbw = mp.width * 0.55
    cax = fig.add_axes([mp.x0 + (mp.width - cbw) / 2,
                        mp.y1 + 0.03, cbw, 0.020])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    cb.set_label('Spatial Density (KDE)', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2.5)
    cax.xaxis.get_offset_text().set_fontsize(6)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # add text box of number of samples
    n_samples = len(df_map) # top righht of map, inside axes
    ax_map.text(0.75, 0.95, f'N = {n_samples:,}', transform=ax_map.transAxes,
                fontsize=8, va='bottom', zorder=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.7, edgecolor='none'))

    ax_map.text(0.04, 0.97, 'a', transform=ax_map.transAxes,
                fontsize=10, va='top', zorder=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.7, edgecolor='none'))

else:
    ax_map = fig.add_subplot(gs[:, 0])
    sc = ax_map.scatter(df_map['longitude'], df_map['latitude'],
                        c=df_map['density'], cmap='YlGnBu_r',
                        s=7, alpha=0.75, edgecolors='none')
    cb = fig.colorbar(sc, ax=ax_map, orientation='horizontal',
                      shrink=0.75, pad=0.04)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    cb.set_label('Spatial Density (KDE)', fontsize=8)
    ax_map.set_xlabel('Longitude (°)')
    ax_map.set_ylabel('Latitude (°N)')
    ax_map.spines['top'].set_visible(False)
    ax_map.spines['right'].set_visible(False)
    ax_map.text(0.04, 0.97, 'a', transform=ax_map.transAxes, fontsize=10, va='top')

# (b) Temporal distribution stacked by month — top right
#
ax_time = fig.add_subplot(gs[0, 1])

years   = sorted(df['year'].unique())
bottom  = np.zeros(len(years))
patches = []

for m in MONTHS_ORDER:
    counts = np.array([((df['year'] == y) & (df['month'] == m)).sum()
                       for y in years])
    ax_time.bar(years, counts, bottom=bottom, color=MONTH_COLORS[m],
                edgecolor='none', width=0.85)
    patches.append(mpatches.Patch(facecolor=MONTH_COLORS[m],
                                   label=MONTH_NAMES[m], edgecolor='none'))
    bottom += counts

ax_time.legend(handles=patches, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.0, 1.01), loc='lower left', ncol=5,
               handlelength=0.9, handleheight=0.8,
               columnspacing=0.5, labelspacing=0.2)
ax_time.set_xlabel('Year', fontsize=10)
ax_time.set_ylabel('Count', fontsize=10)
ax_time.set_xlim(min(years) - 0.8, max(years) + 0.8)
ax_time.set_xticks(np.arange(1985, 2026, 5))
ax_time.tick_params(axis='x', rotation=45, labelsize=8)
ax_time.tick_params(axis='y', labelsize=8)
ax_time.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
ax_time.spines['top'].set_visible(False)
ax_time.spines['right'].set_visible(False)
ax_time.text(0.04, 0.97, 'b', transform=ax_time.transAxes, fontsize=10, va='top')

# (c) SST distribution — middle right

ax_sst = fig.add_subplot(gs[1, 1])

sst_c = df['sst_C'].dropna()
ax_sst.hist(sst_c, bins=50, color='dodgerblue', alpha=0.85,
            edgecolor='k', linewidth=0.4)
ax_sst.axvline(SST_THRESH_C, color=COLOR_VLINE, linewidth=1.5,
               linestyle=':', zorder=5, label='272 K')
ax_sst.legend(fontsize=8, frameon=False, loc='upper right')

ax_sst.set_xlabel('SST (°C)', fontsize=10)
ax_sst.set_ylabel('Count', fontsize=10)
ax_sst.tick_params(labelsize=8)
ax_sst.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
ax_sst.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
ax_sst.spines['top'].set_visible(False)
ax_sst.spines['right'].set_visible(False)
ax_sst.text(0.04, 0.97, 'c', transform=ax_sst.transAxes, fontsize=10, va='top')

# (d) TCWV distribution — bottom right

ax_tcwv = fig.add_subplot(gs[2, 1])

tcwv = df['tcwv_kg_m2'].dropna()
ax_tcwv.hist(tcwv, bins=50, color='#CC2222', alpha=0.85,
             edgecolor='k', linewidth=0.4)

ax_tcwv.set_xlabel('TCWV (kg m$^{-2}$)', fontsize=10)
ax_tcwv.set_ylabel('Count', fontsize=10)
ax_tcwv.tick_params(labelsize=8)
ax_tcwv.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
ax_tcwv.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
ax_tcwv.spines['top'].set_visible(False)
ax_tcwv.spines['right'].set_visible(False)
ax_tcwv.text(0.04, 0.97, 'd', transform=ax_tcwv.transAxes, fontsize=10, va='top')


OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUTPUT_FIG}')
plt.close(fig)
