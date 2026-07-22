"""
10_preprocess_historical.py
============================
Historical BARRA-R2 (1985-2014, 30 years) ko future_5channels.npy jaisa
hi 5-channel format mein convert karta hai, taake autoencoder ko
training ke liye bohat zyada "normal day" examples mil sakein
(10,950+ din, future ke sirf 365 din ke muqable).

Mapping (future channel <- historical source):
  tasmax  <- tasmaxAdjust
  pr      <- prAdjust
  hurs    <- mean(hursminAdjust, hursmaxAdjust)
  rsds    <- rsdsAdjust
  sfcWind <- sfcWindmaxAdjust

Same coarsening factors use kiye hain jo 01_preprocess.py mein the
(LAT_FACTOR=5, LON_FACTOR=7), taake grid shape future data se match kare.

Requirements: pip install xarray netCDF4 dask --user
"""

import numpy as np
import xarray as xr
import os
import gc
import time
from pathlib import Path

# ---- CONFIG ----
BASE_DIR = r"C:\ClimateData\BARRA-R2"
SAVE_DIR = r"C:\Users\Tmart\Desktop\cy"
SAVE_PATH = os.path.join(SAVE_DIR, "historical_5channels.npy")
DATES_SAVE_PATH = os.path.join(SAVE_DIR, "historical_dates.npy")

LAT_FACTOR = 5
LON_FACTOR = 7

# future channel order MUST match 01_preprocess.py: tasmax, pr, hurs, rsds, sfcWind
CHANNEL_ORDER = ["tasmax", "pr", "hurs", "rsds", "sfcWind"]

# historical folder(s) needed per channel
SOURCE_MAP = {
    "tasmax": ["tasmaxAdjust"],
    "pr": ["prAdjust"],
    "hurs": ["hursminAdjust", "hursmaxAdjust"],  # will be averaged
    "rsds": ["rsdsAdjust"],
    "sfcWind": ["sfcWindmaxAdjust"],
}


def load_coarsen_year(filepath):
    """Open one yearly NetCDF file, coarsen spatially, return (days, H, W) float32 array + dates."""
    ds = xr.open_dataset(filepath)
    actual_var = list(ds.data_vars)[0]

    coarsened = ds[actual_var].coarsen(
        lat=LAT_FACTOR, lon=LON_FACTOR, boundary="trim"
    ).mean()
    arr = coarsened.values.astype(np.float32)

    dates = ds["time"].values.copy()
    ds.close()
    return arr, dates


def build_channel(channel_name):
    """Build the full 1985-2014 time series for one channel (concatenating all years)."""
    source_folders = SOURCE_MAP[channel_name]
    print(f"\n📦 Building channel '{channel_name}' from {source_folders}...")

    # Case 1: single source folder (tasmax, pr, rsds, sfcWind)
    if len(source_folders) == 1:
        folder = os.path.join(BASE_DIR, source_folders[0])
        nc_files = sorted(Path(folder).glob("*.nc"))
        yearly_arrays = []
        all_dates = []

        for f in nc_files:
            arr, dates = load_coarsen_year(str(f))
            yearly_arrays.append(arr)
            all_dates.append(dates)
            print(f"   {f.name}: shape={arr.shape}")
            gc.collect()

        full_arr = np.concatenate(yearly_arrays, axis=0)
        full_dates = np.concatenate(all_dates, axis=0)
        return full_arr, full_dates

    # Case 2: hurs -> average of hursmin and hursmax
    else:
        folder_min = os.path.join(BASE_DIR, source_folders[0])
        folder_max = os.path.join(BASE_DIR, source_folders[1])
        files_min = sorted(Path(folder_min).glob("*.nc"))
        files_max = sorted(Path(folder_max).glob("*.nc"))

        assert len(files_min) == len(files_max), "hursmin/hursmax file count mismatch!"

        yearly_arrays = []
        all_dates = []
        for f_min, f_max in zip(files_min, files_max):
            arr_min, dates_min = load_coarsen_year(str(f_min))
            arr_max, dates_max = load_coarsen_year(str(f_max))
            avg = ((arr_min + arr_max) / 2.0).astype(np.float32)
            yearly_arrays.append(avg)
            all_dates.append(dates_min)  # dates should be identical for min/max
            print(f"   {f_min.name} + {f_max.name} -> averaged, shape={avg.shape}")
            gc.collect()

        full_arr = np.concatenate(yearly_arrays, axis=0)
        full_dates = np.concatenate(all_dates, axis=0)
        return full_arr, full_dates


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    t0 = time.time()

    channel_arrays = []
    dates_ref = None

    for ch in CHANNEL_ORDER:
        arr, dates = build_channel(ch)
        channel_arrays.append(arr)
        if dates_ref is None:
            dates_ref = dates
        else:
            # Sanity check: all channels should have the same number of days
            if len(dates) != len(dates_ref):
                print(f"   WARNING: '{ch}' has {len(dates)} days, expected {len(dates_ref)}")

    # Stack into (days, 5, H, W), same layout as future_5channels.npy
    print("\n🔧 Stacking channels...")
    data_historical = np.stack(channel_arrays, axis=1)
    print(f"✅ Final shape: {data_historical.shape}")
    print(f"   Memory: {data_historical.nbytes / 1024**2:.1f} MB")

    print("\n📊 Channel value ranges:")
    for i, var in enumerate(CHANNEL_ORDER):
        ch_data = data_historical[:, i, :, :]
        print(f"   {var}: {ch_data.min():.2f} to {ch_data.max():.2f}")

    np.save(SAVE_PATH, data_historical)
    np.save(DATES_SAVE_PATH, dates_ref)

    elapsed = time.time() - t0
    print(f"\n💾 Saved: {SAVE_PATH}")
    print(f"   File size: {os.path.getsize(SAVE_PATH) / 1024**2:.1f} MB")
    print(f"   Dates saved: {DATES_SAVE_PATH}")
    print(f"\n⏱️  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
