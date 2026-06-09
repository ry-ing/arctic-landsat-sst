#!/usr/bin/env python

"""
This script processes MERRA-2 atmospheric profiles, creates input files for LibRadtran,
runs the radiative transfer simulations, and extracts simulated brightness temperatures
for Landsat thermal bands. The results are saved in a CSV file for later analysis.

libradtran is a fortran-based radiative transfer model that can simulate the transfer of 
radiation through the atmosphere, accounting for absorption, scattering, and emission processes. 
It is available from https://www.libradtran.org/doku.php?id=download

libradtran references:

C. Emde, R. Buras-Schnell, A. Kylling, B. Mayer, J. Gasteiger, U. Hamann, J. Kylling, 
B. Richter, C. Pause, T. Dowling, and L. Bugliaro. The libradtran software package for 
radiative transfer calculations (version 2.0.1). Geoscientific Model Development, 
9(5):1647-1672, 2016.

B. Mayer and A. Kylling. Technical note: The libRadtran software package for radiative 
transfer calculations - description and examples of use. Atmos. Chem. Phys., 
5: 1855-1877, 2005.

Inputs:
- MERRA-2 NetCDF profile files (pressure, temperature, humidity)
- Spectral Response Function (SRF) files for Landsat sensors  
  ( downloaded from USGS: https://landsat.usgs.gov/spectral-characteristics-viewer)

Outputs:
- LibRadtran input files (.inp)
- LibRadtran output files (.out)
- Final CSV with simulated brightness temperatures for each sample and sensor

Libradtran model choices: (Libradtran handbook: https://www.libradtran.org/doc/libRadtran.pdf)
- Radiative Transfer Model: DISORT
- uvspec settings:
- Source: Thermal emission from the surface (SST skin temperature)
- Atmosphere: MERRA-2 profile (p, T, H2O) (Radiosonde) overlaid on a polar background atmosphere (e.g. 'subarctic_summer')
- Surface: Water with emissivity ~0.99 (band-averaged for each sensor)
- Geometry: Nadir-viewing satellite (phi=0, umu=1, sza=0)
- Absorption: Reptran medium resolution 

- Outputs: Spectral radiance (W/(m^2 * sr * nm)) as a function of wavelength, 
    which is then convolved with the Landsat SRF to get in-band radiance and 
    converted to brightness temperature using the Planck inversion formula.

"""

import os
import re
import sys
import time
import subprocess
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 1. Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Inputs ---
MERRA2_PROFILE_DIR = REPO_ROOT / "data/MERRA_2_atmos_profiles"

SRF_TM_FILE        = REPO_ROOT / "data/libradtran/landsat-spectral-info/spectral_b6.txt"
SRF_TIRS_FILE_l8   = REPO_ROOT / "data/libradtran/landsat-spectral-info/landsat-TIRS_SRF_landsat8.csv"
SRF_TIRS1_FILE_l9  = REPO_ROOT / "data/libradtran/landsat-spectral-info/landsat-TIRS1_SRF_landsat9.csv"
SRF_TIRS2_FILE_l9  = REPO_ROOT / "data/libradtran/landsat-spectral-info/landsat-TIRS2_SRF_landsat9.csv"

# --- libRadtran paths --- UPDATE THESE to your local libRadtran 2.0.6 installation ---
LIBRADTRAN_ROOT_DIR = Path("/path/to/libRadtran-2.0.6/data/")
LIBRADTRAN_PATH     = Path("/path/to/libRadtran-2.0.6/bin/uvspec")

# --- Working directories ---
SIM_WORKING_DIR = REPO_ROOT / "data/libradtran_sims"
SIM_ATM_DIR     = SIM_WORKING_DIR / "atm_profiles"
SIM_INP_DIR     = SIM_WORKING_DIR / "inp_files"
SIM_OUT_DIR     = SIM_WORKING_DIR / "out_files"

# --- Output ---
SIMULATION_RESULTS_CSV = REPO_ROOT / "data/libradtran_sims/libradtran_simulations_df_MERRA.csv"

# --- RTM settings (wavelength range in nm) ---
RTM_WAVELENGTH_START_NM = 9800    # 9.8 μm
RTM_WAVELENGTH_END_NM   = 13000   # 13.0 μm

# The MERRA-2 profile overlays the troposphere; this just provides polar
# stratospheric p,T and polar trace-gas columns (O3, CO2, CH4, N2O...).
BACKGROUND_ATMOSPHERE = "subarctic_summer"

# --- UCSB seawater emissivity spectrum ---
# Per-sensor band-averaged emissivity is computed by convolving this
# seawatr1.spc: 10-set average, UCSB.
UCSB_EMISSIVITY_FILE   = REPO_ROOT / "data/libradtran/MODIS_UCSB_seawater1_emissivity.txt"
EMISSIVITY_OUTPUT_FILE = REPO_ROOT / "data/algorithm_params/sensor_emissivity.txt"

# --- Sensor K1/K2 constants for BT inversion (Collection 2) ---
# Units: K1 in W/(m^2·sr·μm), K2 in K.  From USGS Landsat handbooks.
LANDSAT_K_CONSTANTS = {
    'L4': {'B6':  {'K1': 671.62,   'K2': 1284.30}},
    'L5': {'B6':  {'K1': 607.76,   'K2': 1260.56}},
    'L7': {'B6':  {'K1': 666.09,   'K2': 1282.71}},
    'L8': {'B10': {'K1': 774.8853, 'K2': 1321.0789}},
    'L9': {'B10': {'K1': 799.0284, 'K2': 1329.2405}},
}

UVSPEC_TIMEOUT_SECONDS = 60   # libRadtran per run timeout

# ============================================================================
# 2. Helpers
# ============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(REPO_ROOT / "logs/libradtran_simulations.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def check_paths():
    """checks that input files and libRadtran executable exist before starting the simulations."""
    critical = [
        (MERRA2_PROFILE_DIR, "MERRA-2 profile directory"),
        (SRF_TM_FILE,        "TM SRF file (L4/L5/L7)"),
        (SRF_TIRS_FILE_l8,   "TIRS SRF file (L8)"),
        (SRF_TIRS1_FILE_l9,  "TIRS1 SRF file (L9)"),
        (Path(LIBRADTRAN_PATH), "uvspec executable"),
    ]
    for path, description in critical:
        if not Path(path).exists():
            logging.error(f"MISSING: {description} at {path}")
            sys.exit(1)

def get_sensor_from_id(landsat_id: str) -> Optional[str]:
    if pd.isna(landsat_id):
        return None
    if landsat_id.startswith("LC09"): return "L9"
    if landsat_id.startswith("LC08"): return "L8"
    if landsat_id.startswith("LE07"): return "L7"
    if landsat_id.startswith("LT05"): return "L5"
    if landsat_id.startswith("LT04"): return "L4"
    return None

# ============================================================================
# 3. Spectral response functions
# ============================================================================

def load_srf_data() -> dict:
    """Load SRF CSVs for all sensors into a dict keyed by sensor name.
    Wavelenghts are converted to nm here (the TM file comes in um)."""
    srf_data = {}

    # --- L4, L5, L7 (TM / ETM Band 6) ---
    try:
        df_tm = pd.read_csv(
            SRF_TM_FILE,
            delim_whitespace=True,
            skiprows=4,
            names=['Wavelength', 'L7 ETM+', 'L4 TM', 'L5 TM'],
        )
        df_tm['Wavelength_nm'] = df_tm['Wavelength'] * 1000.0

        srf_data['L4'] = df_tm[['Wavelength_nm', 'L4 TM']].rename(columns={'L4 TM': 'Response'})
        srf_data['L5'] = df_tm[['Wavelength_nm', 'L5 TM']].rename(columns={'L5 TM': 'Response'})
        srf_data['L7'] = df_tm[['Wavelength_nm', 'L7 ETM+']].rename(columns={'L7 ETM+': 'Response'})

        logging.info(f"Loaded TM/ETM+ SRF from {SRF_TM_FILE}")
    except Exception as e:
        logging.error(f"Failed loading L4/L5/L7 SRF: {e}")
        raise

    # --- L8 (TIRS Band 10) ---
    try:
        df_tirs = pd.read_csv(SRF_TIRS_FILE_l8)
        df_tirs.columns = [c.strip() for c in df_tirs.columns]
        df_tirs['Wavelength_nm'] = df_tirs['wavelength [um]'] * 1000.0

        srf_data['L8'] = df_tirs[['Wavelength_nm', 'TIRS1 10.8um band average']].rename(
            columns={'TIRS1 10.8um band average': 'Response'}
        )
        logging.info(f"Loaded TIRS SRF (L8) from {SRF_TIRS_FILE_l8}")
    except Exception as e:
        logging.error(f"Failed loading L8 SRF: {e}")
        raise

    # --- L9 (TIRS1 Band 10) ---
    try:
        df_l9 = pd.read_csv(SRF_TIRS1_FILE_l9)
        df_l9.columns = [c.strip() for c in df_l9.columns]
        df_l9['Wavelength_nm'] = df_l9['wavelength [um]'] * 1000.0

        srf_data['L9'] = df_l9[['Wavelength_nm', 'Band 10 Band=Average RSR']].rename(
            columns={'Band 10 Band=Average RSR': 'Response'}
        )
        logging.info(f"Loaded TIRS1 SRF (L9) from {SRF_TIRS1_FILE_l9}")
    except Exception as e:
        logging.error(f"Failed loading L9 SRF: {e}")
        raise

    # --- Clean SRFs ---
    for sensor, df in srf_data.items():
        df['Wavelength_nm'] = pd.to_numeric(df['Wavelength_nm'], errors='coerce')
        df['Response']      = pd.to_numeric(df['Response'],      errors='coerce')
        df = df.dropna().sort_values('Wavelength_nm').reset_index(drop=True)
        df.loc[df['Response'] < 0, 'Response'] = 0.0
        srf_data[sensor] = df

    # Save a diagnostic plot of SRFs
    Path("landsat-spectral-info").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    for sensor, df in srf_data.items():
        plt.plot(df['Wavelength_nm'], df['Response'], label=sensor)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Response")
    plt.title("Landsat spectral response functions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("/data/libradtran_sims/srf_functions.png", dpi=150, bbox_inches='tight')
    plt.close()

    return srf_data

def compute_band_emissivity(srf_data: dict) -> dict:
    """
    SRF-weighted band-average seawater emissivity for each sensor.
    SRF wavelenghts are in nm internally, converted to μm here for interpolation.
    """
    em = pd.read_csv(
        UCSB_EMISSIVITY_FILE, comment='#', delim_whitespace=True,
        names=['wavelength_um', 'wavenumber_cm1', 'emissivity'],
    )
    em = em.dropna().sort_values('wavelength_um').reset_index(drop=True)
    f_em = interp1d(em['wavelength_um'], em['emissivity'],
                    kind='linear', bounds_error=False, fill_value=np.nan)
    result = {}
    for sensor, df in srf_data.items():
        wav_um  = df['Wavelength_nm'].values / 1000.0   # nm -> μm
        resp    = df['Response'].values
        em_vals = f_em(wav_um)
        valid   = (resp > 0) & np.isfinite(em_vals)
        eps = (integrate.trapezoid(em_vals[valid] * resp[valid], wav_um[valid])
               / integrate.trapezoid(resp[valid], wav_um[valid]))
        result[sensor] = float(eps)
        logging.info(f"  {sensor}: band-averaged ε = {eps:.4f}")
    return result

# ============================================================================
# 4. Atmosphere file construction  (MMR instead of RH)
# ============================================================================

def _saturation_vapour_pressure_hPa(T_K: np.ndarray) -> np.ndarray:
    """
    Saturation vapour pressure (hPa) as a function of T (K).

    Bolton (1980) above 0°C, Murphy & Koop (2005) over ice below 0°C.
    """
    T_K = np.asarray(T_K, dtype=float)
    T_C = T_K - 273.15

    # Bolton 1980, over liquid water
    es_water = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))

    # Murphy & Koop 2005, over ice.  Valid for T > 110 K.
    #   ln(es_ice) [Pa] = 9.550426 − 5723.265/T + 3.53068·ln(T) − 0.00728332·T
    # Clip T to a safe lower bound to avoid log of negative
    T_safe = np.maximum(T_K, 150.0)
    es_ice_Pa = np.exp(9.550426
                        - 5723.265 / T_safe
                        + 3.53068  * np.log(T_safe)
                        - 0.00728332 * T_safe)
    es_ice = es_ice_Pa / 100.0   # Pa -> hPa

    return np.where(T_K >= 273.15, es_water, es_ice)


def create_atmosphere_file(ds_profile: xr.Dataset,
                           ds_sl: xr.Dataset,
                           ds_tcwv: xr.Dataset,
                           atm_profile_path: Path):
    """
    Writes a libRadtran radiosonde file (p hPa, T K, RH %) from a MERRA-2 profile
    and returns (sst_skin_k, tcwv_kg_m2) for that sample.

    H2O is passed as RH % to match the 'H2O RH' radiosonde keyword in uvspec.
    Sat. vapour pressure uses Bolton (1980) above 0°C and Murphy & Koop (2005)
    over ice below -- see _saturation_vapour_pressure_hPa. 

    MERRA-2 QV is specific humidity kg/kg. Conversion:
      w = QV / (1-QV),  w_s = 0.622 * e_s / (p - e_s),  RH = 100 * w/w_s.
    """
    # skin T from the MERRA-2 single-level ocean skin product
    sst_skin_k = float(ds_sl['TSKINWTR'].values.item())

    # pull profile variables into a dataframe
    df = pd.DataFrame({
        'pressure_hPa':  ds_profile['lev'].values,
        'temperature_K': ds_profile['T'].values,
        'qv_kgkg':       ds_profile['QV'].values,
    })

    # drop fill values and below-surface levels (-999, -9999, 1e15 etc)
    df = df.replace([-999.0, -9999.0, 1e15], np.nan).dropna()
    if df.empty:
        raise ValueError("MERRA-2 profile is all fill values.")

    # basic range check 
    df = df[(df['pressure_hPa'] > 0) & (df['pressure_hPa'] <= 1050)]
    df = df[(df['temperature_K'] > 150) & (df['temperature_K'] < 330)]
    df = df[df['qv_kgkg'] >= 0]
    if df.empty:
        raise ValueError("MERRA-2 profile has no physically reasonable levels.")

    # convert MERRA-2 specific humidity QV (kg/kg)t o RH %
    
    e_s = _saturation_vapour_pressure_hPa(df['temperature_K'].values)    # hPa
    w_s = 0.622 * e_s / (df['pressure_hPa'].values - e_s)   # sat. mixing ratio
    w   = df['qv_kgkg'].values / (1.0 - df['qv_kgkg'].values)   # actual mixing ratio
    df['RH_percent'] = np.clip(100.0 * w / w_s, 0.0, 100.0)

    # sort ascending pressure (TOA first -- libRadtran convention)
    df = df.sort_values('pressure_hPa', ascending=True).reset_index(drop=True)

    # write 3-col radiosonde file: p(hPa), T(K), RH(%)
    df_out = df[['pressure_hPa', 'temperature_K', 'RH_percent']]
    df_out.to_csv(
        atm_profile_path,
        sep=' ',
        header=False,
        index=False,
        float_format='%.6e',
    )

    # TCWV comes straight from the MERRA-2 SL product (TQV variable)
    tcwv_kg_m2 = float(np.asarray(ds_tcwv['TQV'].values).item())

    return sst_skin_k, tcwv_kg_m2

# ============================================================================
# 5. uvspec input file construction
# ============================================================================

def create_uvspec_input(atm_profile_path: Path,
                         uvspec_inp_path: Path,
                         sst_skin: float,
                         sensor_emissivity: float = 0.9926):
    """
    Write a uvspec .inp file for one atmospheric profile.

    Settings: subarctic_summer background atm, MERRA-2 radiosonde overlay
    (H2O as RH %), reptran medium absorbtion, DISORT solver, nadir geometry
    (sza=0, umu=1.0). Surface emissivty via albedo = 1-emissivity.
    Wavelength range 9800-13000 nm covers all Landsat TIR bands with margin.
    """
    albedo = 1.0 - sensor_emissivity

    input_content = (
        # Data paths
        f"data_files_path {LIBRADTRAN_ROOT_DIR}\n"
        # Polar background atmosphere — MERRA-2 overlays the troposphere
        f"atmosphere_file {BACKGROUND_ATMOSPHERE}\n"
        # MERRA-2 profile: p, T, H2O in RH (%)
        f"radiosonde {atm_profile_path} H2O RH\n"
        # Source
        f"source thermal\n"
        f"sur_temperature {sst_skin:.2f}\n"
        # Absorption parameterisation
        f"mol_abs_param reptran medium\n"
        # Solver & geometry
        f"rte_solver disort\n"
        f"zout TOA\n"
        f"altitude 0.0\n"
        # Surface (Lambertian thermal albedo = 1 − ε)
        f"albedo {albedo:.4f}\n"
        # Wavelength window
        f"wavelength {RTM_WAVELENGTH_START_NM} {RTM_WAVELENGTH_END_NM}\n"
        # Output
        f"output_user lambda uu\n"
        f"output_process per_nm\n"
        # Nadir viewing geometry
        f"phi 0.0\n"
        f"umu 1.0\n"
        f"sza 0.0\n"
        f"quiet\n"
    )
    with open(uvspec_inp_path, 'w') as f:
        f.write(input_content)

def run_libradtran(uvspec_inp_path: Path, uvspec_out_path: Path) -> bool:
    """Run uvspec on one input file, piping stdout to the .out file.
    Returns False if it crashes or hits the timeout."""
    cmd = f"{LIBRADTRAN_PATH} < {uvspec_inp_path} > {uvspec_out_path}"
    try:
        subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=UVSPEC_TIMEOUT_SECONDS,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"uvspec failed for {uvspec_inp_path.name}: {e.stderr.decode()[:500]}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"uvspec TIMEOUT for {uvspec_inp_path.name}")
        return False

# ============================================================================
# 6. Read uvspec output
# ============================================================================

def read_libradtran_output(uvspec_out_path: Path) -> Optional[pd.DataFrame]:
    """Parse uvspec output into a 2-col DataFrame (wavelength_nm, radiance_W_m2_sr_nm).
    Units are already W/(m2·sr·nm) because we use 'output_process per_nm' -- no conversion needed."""
    try:
        df = pd.read_csv(
            uvspec_out_path,
            delim_whitespace=True,
            header=None,
            names=['wavelength_nm', 'radiance_W_m2_sr_nm'],
            on_bad_lines='skip',
        )
    except pd.errors.EmptyDataError:
        logging.warning(f"Empty uvspec output: {uvspec_out_path.name}")
        return None
    except Exception as e:
        logging.error(f"Failed to read {uvspec_out_path.name}: {e}")
        return None

    df = df.dropna()
    if df.empty:
        return None
    return df

# ============================================================================
# 7. SRF convolution and Planck inversion
# ============================================================================

def convolve_srf(lib_output_df: pd.DataFrame, srf_df: pd.DataFrame) -> float:
    """Convolve libRadtran TOA radiance with a sensor SRF to get in-band radiance (W/m2/sr/nm).
    SRF is interpolated onto the libRadtran wavelength grid before integration."""
    lib_wav = lib_output_df['wavelength_nm'].values
    lib_rad = lib_output_df['radiance_W_m2_sr_nm'].values

    srf_wav = srf_df['Wavelength_nm'].values
    srf_res = srf_df['Response'].values

    # Interpolate SRF onto libRadtran wavelength grid
    f_srf = interp1d(srf_wav, srf_res, kind='linear',
                     bounds_error=False, fill_value=0.0)
    srf_resampled = f_srf(lib_wav)

    valid = srf_resampled > 0
    if valid.sum() == 0:
        return float('nan')

    w = lib_wav[valid]
    L = lib_rad[valid]
    R = srf_resampled[valid]

    num = integrate.trapezoid(L * R, w)
    den = integrate.trapezoid(R, w)
    if den == 0:
        return float('nan')

    return num / den

def convert_to_bt(in_band_radiance_nm: float, sensor: str) -> float:
    """Convert in-band radiance to BT (K) using Planck inversion: BT = K2 / ln(K1/L + 1).
    K1/K2 are from the USGS Landsat handbook (per μm), so multiply radiance by 1000 first."""
    band = 'B10' if sensor in ('L8', 'L9') else 'B6'
    try:
        K1 = LANDSAT_K_CONSTANTS[sensor][band]['K1']
        K2 = LANDSAT_K_CONSTANTS[sensor][band]['K2']
    except KeyError:
        logging.error(f"No K-constants for {sensor} {band}.")
        return float('nan')

    L_um = in_band_radiance_nm * 1000.0    # per nm → per μm
    if L_um <= 0 or np.isnan(L_um):
        return float('nan')

    return K2 / np.log((K1 / L_um) + 1.0)

# ============================================================================
# 8. Main
# ============================================================================

def main():
    setup_logging()
    check_paths()

    logging.info("=" * 68)
    logging.info("libRadtran simulation pipeline (v2)")
    logging.info(f"Background atmosphere: {BACKGROUND_ATMOSPHERE}")
    logging.info(f"H2O input format: MMR (g/kg)")
    logging.info("Per-sensor emissivities will be computed from UCSB seawater spectrum.")
    logging.info("=" * 68)

    # 1. Create working directories
    for d in (SIM_ATM_DIR, SIM_INP_DIR, SIM_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load SRFs
    try:
        srf_data = load_srf_data()
    except Exception as e:
        logging.error(f"SRF load failed: {e}")
        return
    sensors_to_run = sorted(srf_data.keys())
    logging.info(f"Will simulate for sensors: {sensors_to_run}")

    # compute per-sensor emissivity from UCSB spectrum and save to file
    logging.info("--- Computing band-averaged emissivity ---")
    emissivity = compute_band_emissivity(srf_data)
    EMISSIVITY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMISSIVITY_OUTPUT_FILE, 'w') as f:
        f.write("# Band-averaged seawater emissivity per Landsat sensor\n")
        f.write("# Source: MODIS UCSB Emissivity Library, seawatr1.spc (10-set average)\n")
        f.write("# Method: eps_band = integral(eps(lam)*R(lam) dlam) / integral(R(lam) dlam)\n")
        f.write("# Consistent with Vanhellemont (2020) Table 1 (L8 B10=0.9926, B11=0.9877)\n")
        f.write(f"# {'Sensor':<8} {'Emissivity'}\n")
        for sensor, eps in sorted(emissivity.items()):
            f.write(f"  {sensor:<8} {eps:.4f}\n")
    logging.info(f"Saved emissivity values to {EMISSIVITY_OUTPUT_FILE}")

    # 3. Find profile files
    profile_files = sorted(MERRA2_PROFILE_DIR.glob("merra2_profile_*.nc"))
    if not profile_files:
        logging.error(f"No profiles found in {MERRA2_PROFILE_DIR}")
        return
    logging.info(f"Found {len(profile_files)} profiles.")

    # 4. Resume support: check which sample_ids are already in the output CSV
    already_done: set = set()
    if SIMULATION_RESULTS_CSV.exists():
        try:
            existing = pd.read_csv(SIMULATION_RESULTS_CSV, usecols=['sample_id'])
            already_done = set(existing['sample_id'].astype(str).unique())
            logging.info(f"Found {len(already_done)} already-simulated samples; will skip.")
        except Exception as e:
            logging.warning(f"Could not read existing results: {e}")

    # 5. Simulation loop
    results_list = []

    for profile_path in tqdm(profile_files, desc="Running simulations"):
        match = re.search(r'merra2_profile_(.+)\.nc', profile_path.name)
        if not match:
            logging.warning(f"Unexpected filename: {profile_path.name}")
            continue

        sample_id = match.group(1)
        if sample_id in already_done:
            continue

        # file paths -- inp/out named per (sample_id, sensor) for per-sensor emissivity runs
        sl_path   = MERRA2_PROFILE_DIR / f"merra2_sst_{sample_id}.nc"
        tcwv_path = MERRA2_PROFILE_DIR / f"merra2_tcvc_{sample_id}.nc"
        atm_path  = SIM_ATM_DIR / f"atm_profile_{sample_id}.dat"

        if not sl_path.exists() or not tcwv_path.exists():
            logging.warning(f"Missing SL or TCWV file for {sample_id}; skipping.")
            continue

        # Open NetCDF files
        try:
            ds_profile = xr.open_dataset(profile_path)
            ds_sl      = xr.open_dataset(sl_path)
            ds_tcwv    = xr.open_dataset(tcwv_path)
        except Exception as e:
            logging.warning(f"NetCDF open failed for {sample_id}: {e}")
            continue

        try:
            # Build atmosphere file and extract skin T / TCWV
            try:
                sst_skin_k, tcwv_kg_m2 = create_atmosphere_file(
                    ds_profile, ds_sl, ds_tcwv, atm_path
                )
            except Exception as e:
                logging.error(f"{sample_id}: atm file build failed: {e}")
                continue

            # one uvspec run per sensor using that sensor's emissivity --
            # emissivties range from L5: 0.9899 to L8: 0.9930, so each sensor
            # gets its own simulation rather than sharing one representative run.
            sim_bts = {}
            any_success = False
            for sensor in sensors_to_run:
                inp_path_s = SIM_INP_DIR / f"uvspec_{sample_id}_{sensor}.inp"
                out_path_s = SIM_OUT_DIR / f"uvspec_{sample_id}_{sensor}.out"

                create_uvspec_input(atm_path, inp_path_s, sst_skin_k, emissivity[sensor])

                if not run_libradtran(inp_path_s, out_path_s):
                    sim_bts[f"sim_bt_{sensor}"] = float('nan')
                    continue

                lib_df = read_libradtran_output(out_path_s)
                if lib_df is None:
                    sim_bts[f"sim_bt_{sensor}"] = float('nan')
                    continue

                in_band_L = convolve_srf(lib_df, srf_data[sensor])
                sim_bts[f"sim_bt_{sensor}"] = convert_to_bt(in_band_L, sensor)
                any_success = True

            if not any_success:
                continue

            result = {
                'sample_id':        sample_id,
                'sst_skin_merra2_k': sst_skin_k,
                'tcwv_merra2_kg_m2': tcwv_kg_m2,
            }
            result.update(sim_bts)
            results_list.append(result)

            bt_str = ", ".join(
                f"{s}:{sim_bts[f'sim_bt_{s}']:.2f}K" for s in sensors_to_run
            )
            logging.info(
                f"{sample_id}  SST={sst_skin_k:.2f}K  TCWV={tcwv_kg_m2:.2f} kg/m²  "
                f"[{bt_str}]"
            )

            # Incremental write every 100 profiles 
            if len(results_list) >= 100:
                df_chunk = pd.DataFrame(results_list)
                df_chunk.to_csv(
                    SIMULATION_RESULTS_CSV,
                    mode='a',
                    header=not SIMULATION_RESULTS_CSV.exists(),
                    index=False,
                )
                results_list = []

        finally:
            ds_profile.close()
            ds_sl.close()
            ds_tcwv.close()

    # 6. Final flush
    if results_list:
        df_chunk = pd.DataFrame(results_list)
        df_chunk.to_csv(
            SIMULATION_RESULTS_CSV,
            mode='a',
            header=not SIMULATION_RESULTS_CSV.exists(),
            index=False,
        )

    logging.info("--- libRadtran pipeline complete ---")
    try:
        final_df = pd.read_csv(SIMULATION_RESULTS_CSV)
        logging.info(f"Total simulations in {SIMULATION_RESULTS_CSV}: {len(final_df)}")
    except Exception:
        pass

if __name__ == "__main__":
    main()