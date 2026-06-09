#!/usr/bin/env python

"""
get_MERRA2-PROFILES_1985-2025.py

Unified script to download MERRA-2 atmospheric profiles for the complete
1985-2025 period

This script generates spatio-temporal sample points over ocean coastal areas
of Southern Greenland and downloads the corresponding MERRA-2 data for each
point.

Uses XARRAY with the PYDAP engine, which is okay for handling NASA Earthdata
authentication. Reads credentials from .netrc file, creates an authenticated
requests.Session, and passes that to xarray.

Downloads THREE files per sample point:
1.  Profile (M2I3NPASM): 3-hourly T, QV, H at pressure levels.
2.  SST (M2T1NXOCN): Hourly TSKINWTR (SST).
3.  TCWV/Cloud (M2T1NXSLV): Hourly TQV (TCWV), TQL (Cloud Liquid), TQI (Cloud Ice).

- .netrc file in your home directory
- EULA "Agreement" for each data collection on the NASA GES DISC website.
- `conda install -c conda-forge pydap`

Note: This consolidates 2000-2025 (9000 samples) and 1985-2000 extension
(5400 samples). To get temporal sampling density across the
40-year period
"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import logging
import time
import random
import calendar
import geopandas as gpd
import netrc
import warnings
import shutil
import netCDF4
import requests
from pydap.client import open_url

# --- 1. CONFIGURATION ---

REPO_ROOT = Path(__file__).resolve().parent.parent

# **OUTPUT:** Directory to save the final profile NetCDF files
MERRA2_OUTPUT_DIR = REPO_ROOT / "data/MERRA_2_atmos_profiles"
MERRA2_OCEAN_MASK_FILE = MERRA2_OUTPUT_DIR / "MERRA-2_land_sea_mask.nc"

# **OUTPUT:** File to store the generated sample points
SAMPLE_POINTS_CSV = REPO_ROOT / "data/MERRA_2_sampling/MERRA-2_sampling_points_1985-2025.csv"
SAMPLE_POINTS_SHP = REPO_ROOT / "data/MERRA_2_sampling/MERRA-2_sample_points_1985-2025.shp"

COASTLINE_SHAPEFILE = Path(r"/path/to/Greenland_coast.shp")

# --- Sampling Parameters ---
GREENLAND_BOUNDS = [75.0, -70.0, 59.0, -16.0]  # [N, W, S, E]
TARGET_PROFILE_COUNT = 14400  # Proportional to 40 years: 9000 * (40/25) = 14,400
YEARS = range(1985, 2025)  # 1985-2025 (40 years, consolidating previous 2000-2025 + 1985-2000 extension)
MONTHS = [5, 6, 7, 8, 9]  # Melt season (May–September)

# --- Coastal Filtering Parameters ---
COASTAL_BUFFER_METERS = 30000  # 30 km
PROJECTED_CRS = "EPSG:3413"

# --- MERRA-2 OPeNDAP Configuration ---
# Profile (3-hourly)
BASE_URL_PROFILE = "dap4://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I3NPASM.5.12.4"
# SST (hourly)
BASE_URL_SST = "dap4://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2T1NXOCN.5.12.4"
# TCWV, Cloud, LandMask (hourly)
BASE_URL_SL = "dap4://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2T1NXSLV.5.12.4"

# We select all levels using a slice
PROFILE_P_LEVEL_SLICE = slice(1000, 0.1)

# Variables to download
PROFILE_VARS = ['T', 'QV', 'H']
SST_VARS = ['TSKINWTR']
TCVC_VARS = ['TQV']

# --- 2. HELPER FUNCTIONS ---
def setup_logging():
    """Configures a simple logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(REPO_ROOT / "logs/get_merra2_profiles_1985-2025.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_netrc_auth():
    """
    Checks for a valid .netrc file for Earthdata and returns credentials.
    Returns:
        (tuple): (username, password) or (None, None) if failed.
    """
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            username, _, password = auth
            logging.info(f"Authentication success: Found credentials for {username} in .netrc file.")
            return username, password
        else:
            logging.error("Authentication failed: No credentials for 'urs.earthdata.nasa.gov' in .netrc file.")
            return None, None
    except (FileNotFoundError, netrc.NetrcParseError) as e:
        if "access too permissive" in str(e):
            logging.error(f"Authentication failed: {e}")
            logging.error("Please run 'chmod 600 ~/.netrc' to fix file permissions.")
        else:
            logging.error(f"Authentication failed: Could not read or parse .netrc file. Error: {e}")
            logging.error("Please create a valid .netrc file in your home directory.")
        return None, None

def get_merra2_collection_id(year: int) -> str:
    """Returns the MERRA-2 collection ID for a given year."""
    if 1980 <= year <= 1991:
        return '100'
    elif 1992 <= year <= 1999:
        return '200'
    elif 2000 <= year <= 2010:
        return '300'
    elif 2011 <= year:
        return '400'
    else:
        raise ValueError(f"Year {year} is outside the MERRA-2 range.")


def download_merra2_ocean_mask(area: list, session) -> bool:
    """
    Downloads the MERRA-2 ocean mask (TSKINWTR) for the specified
    bounding box for a single time step, using the pydap engine.
    """
    if MERRA2_OCEAN_MASK_FILE.exists():
        logging.info("MERRA-2 ocean mask file already exists. Skipping download.")
        return True
        
    logging.info("Downloading MERRA-2 ocean mask (TSKINWTR) for bounding box...")
    
    n, w, s, e = area
    lat_slice = slice(s, n)
    lon_slice = slice(w, e)

    # OPeNDAP URL for just TSKINWTR, for one time, for the box
    # Using TSKINWTR from 1980-01-01 as the mask.
    filename = "MERRA2_100.tavg1_2d_ocn_Nx.19800101.nc4"
    url = f"{BASE_URL_SST}/1980/01/{filename}" 

    try:
        with xr.open_dataset(
            url, 
            engine="pydap", 
            backend_kwargs={"session": session}
        ) as ds:
            subset = ds.sel(
                lat=lat_slice, 
                lon=lon_slice).sel(
                time='1980-01-01 00:30:00',
                method='nearest'
            )
            subset[SST_VARS[0]].to_netcdf(MERRA2_OCEAN_MASK_FILE)
            
        logging.info(f"Successfully downloaded ocean mask: {MERRA2_OCEAN_MASK_FILE}")
        return True
    except Exception as e:
        logging.error(f"Failed to download MERRA-2 ocean mask: {e}")
        return False

def get_ocean_grid_points(mask_file: Path, area: list) -> list:
    """
    Processes the MERRA-2 ocean mask (TSKINWTR) NetCDF file
    to find all ocean points (where TSKINWTR is not NaN).
    """
    logging.info(f"Identifying ocean points (where TSKINWTR is not NaN)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with xr.open_dataset(mask_file) as ds:
            sst_data = ds[SST_VARS[0]].load()
            
    # Find where the data is NOT null. This is our ocean mask.
    ocean_mask = sst_data.notnull()
    
    if ocean_mask.sum() == 0:
        logging.error("No ocean points found (all TSKINWTR values were NaN).")
        return []

    # Get the coordinate values
    all_lats = sst_data.lat.values
    all_lons = sst_data.lon.values
    
    # Get the indices (row, col) of all True values in the mask
    valid_coord_indices = np.argwhere(ocean_mask.values)
    
    # Create a list of (lat, lon) tuples
    ocean_coords = [
        (all_lats[idx[0]], all_lons[idx[1]]) for idx in valid_coord_indices
    ]

    logging.info(f"Found {len(ocean_coords)} valid ocean grid points.")
    return ocean_coords

def filter_points_by_coastline(
    ocean_points_list: list, 
    coastline_shp_path: Path, 
    buffer_m: int
) -> list:
    """
    Filters ocean points to keep only those within a buffer of the coastline.
    """
    if not ocean_points_list:
        logging.warning("No ocean points provided to filter.")
        return []
    
    if not coastline_shp_path.exists():
        logging.error(f"Coastline shapefile not found at: {coastline_shp_path}")
        return []

    logging.info(f"Loading coastline from {coastline_shp_path}...")
    try:
        coast_gdf = gpd.read_file(coastline_shp_path)
        
        if coast_gdf.crs is None:
            logging.warning(f"Coastline shapefile has no CRS. Assuming it is {PROJECTED_CRS}.")
            coast_gdf.crs = PROJECTED_CRS
            
        lons = [lon for lat, lon in ocean_points_list]
        lats = [lat for lat, lon in ocean_points_list]
        ocean_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lons, lats),
            crs="EPSG:4326"
        )

        logging.info(f"Reprojecting both layers to {PROJECTED_CRS} for buffering...")
        coast_projected = coast_gdf.to_crs(PROJECTED_CRS)
        ocean_projected = ocean_gdf.to_crs(PROJECTED_CRS)
        
        logging.info(f"Buffering coastline by {buffer_m} meters...")
        coast_buffer_geom = coast_projected.buffer(buffer_m)
        coast_buffer_gdf = gpd.GeoDataFrame(
            geometry=coast_buffer_geom, crs=PROJECTED_CRS
        )
        
        logging.info("Performing spatial join to find points within buffer...")
        coastal_points_projected = gpd.sjoin(
            ocean_projected, coast_buffer_gdf, how='inner', predicate='within'
        )
        
        valid_indices = coastal_points_projected.index
        final_ocean_points = [ocean_points_list[i] for i in valid_indices]
        
        logging.info(f"Filtered points. Kept {len(final_ocean_points)} coastal ocean points.")
        
        if not final_ocean_points:
             logging.warning("Filtering resulted in 0 points. Check your bounding box and shapefile.")
             
        return final_ocean_points
        
    except Exception as e:
        logging.error(f"Error during coastline filtering: {e}")
        return []

def generate_sampling_dataframe(ocean_points: list, years: range, months: list, count: int) -> pd.DataFrame:
    """
    Generates random spatio-temporal sample points from the ocean_points pool.
    """
    if not ocean_points:
        logging.error("Cannot generate samples: No ocean points provided.")
        return pd.DataFrame()
        
    logging.info(f"Generating {count} random spatio-temporal sample points...")
    data = []
    for _ in tqdm(range(count), desc="Generating Samples"):
        lat, lon = random.choice(ocean_points)
        year = random.choice(list(years))
        month = random.choice(months)
        max_days = calendar.monthrange(year, month)[1]
        day = random.randint(1, max_days)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        
        data.append({
            'latitude': lat,
            'longitude': lon,
            'timestamp': timestamp
        })
        
    df = pd.DataFrame(data)
    logging.info("Sample generation complete.")
    return df

def fetch_merra2_profile_point(
    lat: float, lon: float, dt: pd.Timestamp, output_filename: Path, session
) -> bool:
    """Fetches a single MERRA-2 *profile* file (M2I3NPASM) for a specific point and time."""
    try:
        collection_id = get_merra2_collection_id(dt.year)
        merra_dt = dt.round('3H') 
        
        year_str = f"{merra_dt.year:04d}"
        month_str = f"{merra_dt.month:02d}"
        day_str = f"{merra_dt.day:02d}"
        
        filename = f"MERRA2_{collection_id}.inst3_3d_asm_Np.{year_str}{month_str}{day_str}.nc4"
        url = f"{BASE_URL_PROFILE}/{year_str}/{month_str}/{filename}"

        ds = xr.open_dataset(url, engine="pydap", backend_kwargs={"session": session}).sel(
                lat=lat, 
                lon=lon, 
                time=merra_dt, 
                method='nearest'
            )
        ds[PROFILE_VARS].load().to_netcdf(output_filename)

        return True
        
    except Exception as e:
        logging.error(f"  Failed to download PROFILE for {output_filename.name}: {e}")
        if output_filename.exists():
            output_filename.unlink()
        return False

def fetch_merra2_sst_point(
    lat: float, lon: float, dt: pd.Timestamp, output_filename: Path, session
) -> bool:
    """Fetches a single MERRA-2 *SST* file (M2T1NXOCN) for a specific point and time."""
    try:
        collection_id = get_merra2_collection_id(dt.year)
        
        hour = dt.round('H').hour
        if hour == 0 and dt.minute < 30:
            merra_dt = (dt.replace(hour=0, minute=30) - pd.Timedelta(hours=1))
        else:
            merra_dt = dt.replace(hour=hour, minute=30)
            if dt.minute < 30:
                merra_dt -= pd.Timedelta(hours=1)
        
        year_str = f"{merra_dt.year:04d}"
        month_str = f"{merra_dt.month:02d}"
        day_str = f"{merra_dt.day:02d}"
        
        filename = f"MERRA2_{collection_id}.tavg1_2d_ocn_Nx.{year_str}{month_str}{day_str}.nc4"
        url = f"{BASE_URL_SST}/{year_str}/{month_str}/{filename}"

        with xr.open_dataset(
            url, 
            engine="pydap", 
            backend_kwargs={"session": session}
        ) as ds:
            subset = ds[SST_VARS].sel(
                lat=lat, 
                lon=lon, 
                time=merra_dt, 
                method='nearest'
            )
            subset.load().to_netcdf(output_filename)
        return True
        
    except Exception as e:
        logging.error(f"  Failed to download SST for {output_filename.name}: {e}")
        if output_filename.exists():
            output_filename.unlink()
        return False

def fetch_merra2_tcvc_point(
    lat: float, lon: float, dt: pd.Timestamp, output_filename: Path, session
) -> bool:
    """Fetches a single MERRA-2 *TCWV/Cloud* file (M2T1NXSLV) for a specific point and time."""
    try:
        collection_id = get_merra2_collection_id(dt.year)

        hour = dt.round('H').hour
        if hour == 0 and dt.minute < 30:
            merra_dt = (dt.replace(hour=0, minute=30) - pd.Timedelta(hours=1))
        else:
            merra_dt = dt.replace(hour=hour, minute=30)
            if dt.minute < 30:
                merra_dt -= pd.Timedelta(hours=1)
        
        year_str = f"{merra_dt.year:04d}"
        month_str = f"{merra_dt.month:02d}"
        day_str = f"{merra_dt.day:02d}"
        
        filename = f"MERRA2_{collection_id}.tavg1_2d_slv_Nx.{year_str}{month_str}{day_str}.nc4"
        url = f"{BASE_URL_SL}/{year_str}/{month_str}/{filename}"

        with xr.open_dataset(
            url, 
            engine="pydap", 
            backend_kwargs={"session": session}
        ) as ds:
            subset = ds[TCVC_VARS].sel(
                lat=lat, 
                lon=lon, 
                time=merra_dt, 
                method='nearest'
            )
            subset.load().to_netcdf(output_filename)
        return True
    
    except Exception as e:
        logging.error(f"  Failed to download TCWV/Cloud for {output_filename.name}: {e}")
        if output_filename.exists():
            output_filename.unlink()
        return False


# --- 3. MAIN SCRIPT ---

def main():
    setup_logging()
    logging.info("=" * 70)
    logging.info("MERRA-2 Profile Data Generation: 1985–2025 (consolidated)")
    logging.info(f"Years: {YEARS.start}–{YEARS.stop - 1} ({len(list(YEARS))} years)")
    logging.info(f"Target samples: {TARGET_PROFILE_COUNT}")
    logging.info(f"Melt season months: {MONTHS}")
    logging.info("=" * 70)
    
    # --- 0. AUTHENTICATION ---
    username, password = check_netrc_auth()
    if not username:
        logging.error("Halting script. Please set up your .netrc file for NASA Earthdata.")
        return
        
    logging.info("Creating authenticated requests.Session...")
    try:
        session = requests.Session()
        session.auth = (username, password)
    except Exception as e:
        logging.error(f"Failed to create authenticated Session: {e}")
        return
    
    if not COASTLINE_SHAPEFILE.exists():
        logging.error(f"CRITICAL: Coastline shapefile not found at {COASTLINE_SHAPEFILE}")
        return

    MERRA2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_POINTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # --- Part 1: Generate Sampling Points ---
    
    if SAMPLE_POINTS_CSV.exists():
        logging.info(f"Loading existing sample points from {SAMPLE_POINTS_CSV}...")
        sample_df = pd.read_csv(SAMPLE_POINTS_CSV, parse_dates=['timestamp'])
        if 'sample_id' in sample_df.columns:
            sample_df = sample_df.set_index('sample_id')
            
    else:
        # 1a. Download the ocean mask
        if not download_merra2_ocean_mask(GREENLAND_BOUNDS, session):
            logging.error("Halting script: Could not download MERRA-2 ocean mask.")
            return

        # 1b. Get all ocean coordinates from the mask
        all_ocean_points = get_ocean_grid_points(MERRA2_OCEAN_MASK_FILE, GREENLAND_BOUNDS)
        if not all_ocean_points:
            logging.error("Halting script: Could not retrieve ocean grid points.")
            return

        # 1c. Filter them using the shapefile
        coastal_ocean_points = filter_points_by_coastline(
            all_ocean_points,
            COASTLINE_SHAPEFILE,
            COASTAL_BUFFER_METERS
        )
        if not coastal_ocean_points:
            logging.error("Halting script: Filtering with shapefile resulted in 0 points.")
            return
            
        # 1d. Generate samples from the *filtered* list
        sample_df = generate_sampling_dataframe(
            coastal_ocean_points, 
            YEARS, 
            MONTHS, 
            TARGET_PROFILE_COUNT
        )
        
        if sample_df.empty:
            logging.error("Halting script: Could not generate sampling DataFrame.")
            return
            
        sample_df.to_csv(SAMPLE_POINTS_CSV, index_label='sample_id')
        logging.info(f"Saved {len(sample_df)} sample points to {SAMPLE_POINTS_CSV}")
        
    logging.info(f"Total points to process: {len(sample_df)}")

    # --- Part 1.5: Export Points to Shapefile for Review ---
    if not SAMPLE_POINTS_SHP.exists():
        logging.info(f"Exporting sample points to {SAMPLE_POINTS_SHP}...")
        try:
            gdf = gpd.GeoDataFrame(
                sample_df, 
                geometry=gpd.points_from_xy(sample_df.longitude, sample_df.latitude),
                crs="EPSG:4326"
            )
            gdf.to_file(SAMPLE_POINTS_SHP)
            logging.info("Shapefile export complete.")
        except Exception as e:
            logging.error(f"Failed to export shapefile: {e}")
    
    # --- Part 2: Download MERRA-2 Data for All Sample Points ---
    
    logging.info("Starting download loop... (This will be slow and may take many hours/days)")
    
    for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc="Downloading MERRA-2 Data"):
        
        file_id = f"greenland_sample_{row.Index}"
        profile_nc_path = MERRA2_OUTPUT_DIR / f"merra2_profile_{file_id}.nc"
        sst_nc_path = MERRA2_OUTPUT_DIR / f"merra2_sst_{file_id}.nc"
        tcvc_nc_path = MERRA2_OUTPUT_DIR / f"merra2_tcvc_{file_id}.nc"
        
        # Check if all THREE files exist
        if profile_nc_path.exists() and sst_nc_path.exists() and tcvc_nc_path.exists():
            continue

        sample_time = row.timestamp
        lat = row.latitude
        lon = row.longitude
        
        # --- 1. Fetch Profile (PL) data ---
        if not profile_nc_path.exists():
            pl_success = fetch_merra2_profile_point(
                lat, lon, sample_time, profile_nc_path, session
            )
            if not pl_success:
                logging.warning(f"Failed PL download for {file_id}. Will retry next time.")
                time.sleep(1)
                continue 
        
        # --- 2. Fetch SST data ---
        if (profile_nc_path.exists() and not sst_nc_path.exists()):
            sst_success = fetch_merra2_sst_point(
                lat, lon, sample_time, sst_nc_path, session
            )
            if not sst_success:
                logging.warning(f"Failed SST download for {file_id}. Will retry next time.")
                time.sleep(1)
                continue 
                
        # --- 3. Fetch TCWV/Cloud data ---
        if (profile_nc_path.exists() and sst_nc_path.exists() and not tcvc_nc_path.exists()):
            tcvc_success = fetch_merra2_tcvc_point(
                lat, lon, sample_time, tcvc_nc_path, session
            )
            if not tcvc_success:
                logging.warning(f"Failed TCWV/Cloud download for {file_id}. Will retry next time.")
                time.sleep(1)

        time.sleep(0.5) 

    logging.info("=" * 70)
    logging.info("MERRA-2 Profile Data Download Finished")
    logging.info("=" * 70)

if __name__ == "__main__":
    main()
