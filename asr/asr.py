import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from astropy.visualization import time_support
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm.autonotebook import tqdm
from IPython.display import clear_output, display
import pandas as pd
from scipy.signal import argrelextrema
import time
import os
from decimal import Decimal
import gc

def minmax_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify local minima and maxima in the 'xl' column of the dataframe and return a new dataframe
    with the indices and types ('min' or 'max') of these extrema.

    Parameters:
    df (pd.DataFrame): Input dataframe with a column named 'xl'.

    Returns:
    pd.DataFrame: Dataframe with columns 'idx' and 'type' indicating the indices and types of extrema.

    PSA: This docstring was produced with the aid of AI.
    """
    # Find indices of local minima and maxima
    min_mask = argrelextrema(df.xl.values, np.less_equal, order=1)[0]
    max_mask = argrelextrema(df.xl.values, np.greater_equal, order=1)[0]

    # Create a dictionary with indices as keys and 'min' or 'max' as values
    min_max_dict = {idx: "min" for idx in min_mask}
    min_max_dict.update({idx: "max" for idx in max_mask})

    # Sort the dictionary by index
    min_max_dict = dict(sorted(min_max_dict.items()))

    # Convert the dictionary to a dataframe
    df_idx = pd.DataFrame.from_dict(min_max_dict, orient='index', columns=['type'])
    df_idx = df_idx.reset_index().rename(columns={'index': 'idx'})

    # Check if the first element is a max, drop it
    if df_idx.type[0] == "max":
        df_idx.drop(0, inplace=True)
    
    df_idx.reset_index(drop=True, inplace=True)
    return df_idx


def flare_class(delta_flux: float) -> tuple[str, str]:
    """
    Classify a solar flare based on its delta flux.

    Parameters:
    delta_flux (float): The change in flux of the solar flare in W/m^2.

    Returns:
    tuple[str, str]: A tuple containing the flare class ('A', 'B', 'C', 'M', 'X') and the formatted string.

    PSA: This docstring was produced with the aid of AI.
    """
    # A flare is classified into one of the following classes based on the delta flux:
    # A: delta_flux < 1e-7 W/m^2
    # B: 1e-7 W/m^2 <= delta_flux < 1e-6 W/m^2
    # C: 1e-6 W/m^2 <= delta_flux < 1e-5 W/m^2
    # M: 1e-5 W/m^2 <= delta_flux < 1e-4 W/m^2
    # X: delta_flux >= 1e-4 W/m^2

    # Always write delta_flux in scientific notation
    if delta_flux < 1e-7:
        delta_flux = delta_flux * 1e8
        first = str(delta_flux).split(".")[0]
        second = str(delta_flux).split(".")[1][0]
        return "A", f"A{first}.{second}"
    elif 1e-7 <= delta_flux < 1e-6:
        delta_flux = delta_flux * 1e7
        first = str(delta_flux).split(".")[0]
        second = str(delta_flux).split(".")[1][0]
        return "B", f"B{first}.{second}"
    elif 1e-6 <= delta_flux < 1e-5:
        delta_flux = delta_flux * 1e6
        first = str(delta_flux).split(".")[0]
        second = str(delta_flux).split(".")[1][0]
        return "C", f"C{first}.{second}"
    elif 1e-5 <= delta_flux < 1e-4:
        delta_flux = delta_flux * 1e5
        first = str(delta_flux).split(".")[0]
        second = str(delta_flux).split(".")[1][0]
        return "M", f"M{first}.{second}"
    else:
        delta_flux = delta_flux * 1e4
        first = str(delta_flux).split(".")[0]
        second = str(delta_flux).split(".")[1][0]
        return "X", f"X{first}.{second}"
    


def get_fs_atlas(df: pd.DataFrame, df_idx: pd.DataFrame, codename: str, datapath: str, pref_ftype:str="json") -> pd.DataFrame:
    """
    Analyze a dataframe to identify solar flares and save the results to a JSON file.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns 'xl' and 'time_tag'.
    df_idx (pd.DataFrame): Dataframe with columns 'idx' and 'type' indicating the indices and types of extrema.
    codename (str): Identifier for the source of the data.
    datapath (str): Path to save the resulting JSON file.
    pref_ftype (str): Preferred file type for saving the results. Default is 'json' ;-).

    Returns:
    pd.DataFrame: Dataframe containing information about identified flares.

    PSA: This docstring was produced with the aid of AI.
    """
    # Delete pre-existing files
    if os.path.exists(f"{datapath}f_{codename}.json"):
        os.remove(f"{datapath}f_{codename}.json")

    # Initialize lists to store flare data
    peak_flux = []
    tstart = []
    tpeak = []
    tend = []
    ratio_arr = []
    start_flux = []
    flare_id = []
    flux_int = []
    flux_int_corrected = []
    delta_flux = []
    fclass_simple = []
    fclass_full = []
    discarded = []

    id = 0

    # Loop through the dataframe to find flares
    for j in tqdm(range(len(df_idx) - 2), desc="Finding flares"):
        if df_idx.type[j] == "min" and df_idx.type[j + 1] == "max":
            h = df_idx.idx[j]
            if h != 0:
                min_val = (df.xl[h] + df.xl[h - 1]) / 2
            else:
                min_val = df.xl[h]

            thr = 2e-8
            k = df_idx.idx[j + 1]
            max_val = df.xl[k]  # Value of flux at the peak

            # Check if the peak is above the threshold, if so, continue
            if max_val > thr + min_val:
                ratio = max_val / min_val
                peak_flux.append(df.xl[df_idx.idx[j + 1]])
                tstart.append(df.time_tag[df_idx.idx[j]])
                tpeak.append(df.time_tag[df_idx.idx[j + 1]])
                ratio_arr.append(ratio)
                start_flux.append(df.xl[df_idx.idx[j]])
                delta_flux.append(max_val - min_val)
                fclass_simple.append(flare_class(max_val - min_val)[0])
                fclass_full.append(flare_class(max_val - min_val)[1])

                # Starting from the peak, find the end of the flare by looking for the index where the flux is below the threshold
                while df.xl[k] > min_val + thr:
                    k += 1
                    if k == len(df.xl):
                        k -= 1
                        break

                # If the next minimum is before the end of the flare, set the end of the flare to the next minimum
                if df.time_tag[k] >= df.time_tag[df_idx.idx[j + 2]]:
                    end = df_idx.idx[j + 2]
                else:
                    end = k
                tend.append(df.time_tag[end])

                # Get the flux integral as the integral of the flux between the start and end times, considering the background flux which is the flux at the start time and needs to be removed at each time step
                flux_int_val = np.trapezoid(df.xl[df_idx.idx[j]:end])
                flux_int_corrected_val = np.trapezoid(df.xl[df_idx.idx[j]:end] - df.xl[df_idx.idx[j]])
                if flux_int_val == np.nan or flux_int_corrected_val == np.nan:
                    print("NAN value detected, you might want to consider checking the data")

                flux_int.append(flux_int_val)
                flux_int_corrected.append(flux_int_corrected_val)
            else:
                discarded.append(df.time_tag[df_idx.idx[j + 1]])

    # Handle the ID assignment. If the flare starts at the same time as the previous one ends, assign the same ID. If not, assign a new ID
    for i in tqdm(range(len(tstart)), desc="Assigning flare IDs"):
        if i == 0:
            flare_id.append(f"{codename}--{id}")
        else:
            if tstart[i] != tend[i - 1]:
                id += 1
                flare_id.append(f"{codename}--{id}")
            else:
                flare_id.append(f"{codename}--{id}")

    # Create a dataframe with the flare data
    flares = pd.DataFrame({
        "flare_id": flare_id,
        "tstart": tstart,
        "tpeak": tpeak,
        "tend": tend,
        "peak_flux": peak_flux,
        "BG_flux": start_flux,
        "flux_int": flux_int,
        "ratio": ratio_arr,
        "flux_int_corrected": flux_int_corrected,
        "delta_flux": delta_flux,
        "fclass_simple": fclass_simple,
        "fclass_full": fclass_full
    })

    # Drop flares with NaN values in flux_int or flux_int_corrected
    old_len = len(flares)
    flares.dropna(subset=["flux_int", "flux_int_corrected"], inplace=True)
    new_len = len(flares)

    # Save the dataframe
    if pref_ftype == "csv":
        flares.to_csv(datapath + f"f_{codename}.csv")
    elif pref_ftype == "json":
        flares.to_json(datapath + f"f_{codename}.json")
    else:
        raise ValueError("Invalid file type. Please choose either 'csv' or 'json'.")

    # Clean up and return the dataframe
    del peak_flux, tstart, tpeak, tend, ratio_arr, start_flux, flare_id, flux_int, flux_int_corrected, delta_flux
    gc.collect()
    return flares


def process_flares(datapath: str, years: np.ndarray) -> None:
    """
    Process min and max df and GOES flux series for a range of years and save the results to JSON files.

    Parameters:
    datapath (str): Path to the directory containing the data files and where the results will be saved.
    years (np.ndarray): Array of years to process.

    Returns:
    None

    PSA: This docstring was produced with the aid of AI.
    """
    for year in years:
        start = time.time()
        df = pd.read_csv(f"{datapath}goes_{year}.json")
        df = df.infer_objects(copy=False)
        df = df.reset_index(drop=True)
        df_idx = minmax_df(df)
        flares = get_fs_atlas(df, df_idx, str(year), datapath)
        print(f"Year {year} took {time.time() - start} seconds")



