# Arctic Landsat SST Retrieval

Code for retrieving sea surface temperature from Landsat 4/5/7/8/9 thermal imagery over Arctic coastal waters. The algorithm is trained on libRadtran radiative transfer simulations with by MERRA-2 atmospheric profiles, and validated against EasyCORA in-situ observations (CMEMS).

Ing R, Nienow P, Slater D, Medina-Lopez E. Investigating the use of Landsat-derived sea surface temperatures as a proxy for changes in ocean forcing of Greenland’s marine-terminating outlet glaciers. Journal of Glaciology. 2026;72:e75. doi:10.1017/jog.2026.10204

---

## Pipeline

Scripts are in `code/` and are written to be run in a hpc environment. To pipeline is briefly described below:

**1. Download MERRA-2 atmospheric profiles**  
```
get_MERRA2-PROFILES_1985-2025.py
```
Downloads MERRA-2 reanalysis profiles (temperature, relative humidity) for a grid of Greenland coastal ocean sampling points spanning 1985–2025. Output goes to `data/MERRA_2_atmos_profiles/`. Sampling point locations are in `data/MERRA_2_sampling/`. 

**2. Run libRadtran radiative transfer simulations**  
```
run_LIBRADTRAN-SIMS.py - requires download of libradtran v2.0.6 (http://www.libradtran.org/) 
```
Runs libRadtran for each MERRA-2 profile to simulate Landsat brightness temperatures across the range of Arctic SST and atmospheric conditions. Outputs `.inp`/`.out` files and a compiled results CSV (`data/libradtran_sims/libradtran_simulations_df_MERRA.csv`). Landsat spectral response functions are in `data/libradtran/landsat-spectral-info/` from USGS handbook

**3. Match in-situ observations to Landsat scenes (GEE)**  *run on HPC, very slow and takes weeks*
```
get_INSITU-LANDSAT-PAIRS.py
```
Reads EasyCORA in-situ NetCDF files, filters to near-surface observations (≤3 m depth), and submits batched export tasks to Google Earth Engine to match each observation to the nearest Landsat scene (±3 h, 100 m spatial buffer). Results export as CSVs to Google Drive and are synced locally to `data/GEE_validation_files/`. Processing log is saved to a JSON file so the script can be restarted without re-submitting completed tasks.

**4. Derive SST coefficients and validate** 
```
derive_SST-COEFFS-main.py
```
Fits per-sensor regression coefficients (A, B, C) from the libRadtran simulations, then derives a bias correction term D via 5-fold cross-validation against the in-situ matchups. Outputs algorithm parameters to `data/algorithm_params/` and validation figures to `Figures/`.

---

## Reproducing the coefficient derivation

The libRadtran simulation results and matched in-situ validation data are deposited on Zenodo (https://doi.org/10.5281/zenodo.20607850) so the coefficient derivation (step 4) can be re-run without running the full pipeline. 

---

## Repository structure

```
.
├── code/
│   ├── get_MERRA2-PROFILES_1985-2025.py        # step 1
│   ├── run_LIBRADTRAN-SIMS.py                  # step 2
│   ├── get_INSITU-LANDSAT-PAIRS.py             # step 3
│   ├── derive_SST-COEFFS-main.py               # step 4
│   ├── plot_MERRA2_sampling.py            
│   └── plot_outlier_analysis.py            
│
├── data/
│   ├── algorithm_params/            # Algorithm outputs
│   │   ├── MASTER_ALGORITHM_PARAMETERS.csv
│   │   └── VALIDATION_STATISTICS.csv
│   ├── libradtran/
│   │   └── landsat-spectral-info/   # Landsat SRF files
│   ├── MERRA_2_sampling/            # Sampling point locations (shapefile + CSV)
│   │
│   ├── MERRA_2_atmos_profiles/      
│   ├── libradtran_sims/             
│   └── GEE_validation_files/        
├── Figures/                         # Output plots
└── logs/                            # SLURM and script logs 
```

---

## Large data files

The following are excluded from this repo. The libRadtran simulations and matched validation data are available from Zenodo ([10.5281/zenodo.20607850](https://doi.org/10.5281/zenodo.20607850)). The MERRA-2 profiles must be re-downloaded from NASA.

| Directory | Contents | Source |
|-----------|----------|--------|
| `data/MERRA_2_atmos_profiles/` | MERRA-2 profile NetCDFs  | NASA |
| `data/libradtran_sims/` | `.inp`/`.out` files + compiled CSV  | Zenodo |
| `data/GEE_validation_files/` | Matched in-situ/Landsat CSVs | Zenodo |

The EasyCORA in-situ data must be downloaded separately from [CMEMS](https://marine.copernicus.eu/) (dataset ID: `cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr`).

---

## Dependencies

A conda environment file is provided at `environment.yml`. To create the environment:

- [libRadtran](http://www.libradtran.org/) 2.0.6 
- Google Earth Engine account with Drive export permissions
