from typing import Dict, List, Optional, Union
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries as ts
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from IPython.display import clear_output
import glob
import netCDF4 as nc

def goes_downloader_main(year: int, custom_sat_data: Optional[str] = None, ignore_quality: bool = False, keep_local: bool = True) -> pd.DataFrame:
    """
    Main function to download GOES data for a given year.

    Parameters:
    year (int): The year for which to download data.
    custom_sat_data (Optional[str]): Path to custom satellite data CSV file.
    ignore_quality (bool): Whether to ignore quality flags.
    keep_local (bool): Whether to keep local data.

    Returns:
    pd.DataFrame: DataFrame containing the downloaded data.
    """
    secondary_flag = None
    
    if year < 2002:
        files_primary =  sorted(glob.glob(f"data/downloads/{year}_primary/*.nc"))
        print(f"Found {files_primary[0]} files for year {year} in primary directory.")
        df = legacy_handler_MANUAL(files_primary[0])
        print(f"Finished processing data for {year}")
        return df

    if custom_sat_data is not None:
        satellite_years = pd.read_csv(custom_sat_data)
        satellite_years['start_datetime'] = pd.to_datetime(satellite_years['start_datetime'])
        satellite_years = satellite_years[satellite_years['start_datetime'].dt.year == year]
        satellite_years.reset_index(drop=True, inplace=True)

        if not keep_local:
            # Download the data
            for i in range(len(satellite_years)):
                tstart = satellite_years['start_datetime'][i]
                tend = satellite_years['end_datetime'][i]
                result_primary = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"), a.goes.SatelliteNumber(int(satellite_years['primary'][i])))
                if not np.isnan(satellite_years["secondary"][i]) and not ignore_quality:
                    result_secondary = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"), a.goes.SatelliteNumber(int(satellite_years['secondary'][i])))
                    secondary_flag = True
                if np.isnan(satellite_years["secondary"][i]):
                    print("No secondary satellite data available, falling back to primary satellite data")
                    result_secondary = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"), a.goes.SatelliteNumber(int(satellite_years['primary'][i])))
                    secondary_flag = True
                if not os.path.exists(f"data/downloads/{year}_primary/"):
                    os.makedirs(f"data/downloads/{year}_primary/")
                    if not ignore_quality and secondary_flag:
                        if not os.path.exists(f"data/downloads/{year}_secondary/"):
                            os.makedirs(f"data/downloads/{year}_secondary/")
                files_primary = Fido.fetch(result_primary, path=f"data/downloads/{year}_primary")
                if not ignore_quality and secondary_flag:
                    files_secondary = Fido.fetch(result_secondary, path=f"data/downloads/{year}_secondary")
    else:
        df = goes_downloader_old(year)
        print(f"Finished processing data for {year}")
        return df

    clear_output()


    try:
        files_primary = [f"data/downloads/{year}_primary/{file}" for file in os.listdir(f"data/downloads/{year}_primary/")]
        goes_ts_primary = ts.TimeSeries(files_primary, concatenate=True, allow_errors=True)
        df_primary = goes_ts_primary.to_dataframe()
        df_primary.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
        df_primary.rename(columns={"xrsb":"xl", "xrsb_quality":"xl_quality"}, inplace=True)
        df_primary.reset_index(inplace=True)
        df_primary.rename(columns={"index":"time_tag"}, inplace=True)
    except IndexError:
        print("Index error in primary satellite data, falling back to secondary satellite data")
        try:
            files_primary = [f"data/downloads/{year}_secondary/{file}" for file in os.listdir(f"data/downloads/{year}_secondary/")]
            goes_ts_primary = ts.TimeSeries(files_primary, concatenate=True, allow_errors=True)
            df_primary = goes_ts_primary.to_dataframe()
            df_primary.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
            df_primary.rename(columns={"xrsb":"xl", "xrsb_quality":"xl_quality"}, inplace=True)
            df_primary.reset_index(inplace=True)
            df_primary.rename(columns={"index":"time_tag"}, inplace=True)
            secondary_flag = False
        except IndexError:
            print("Index error in secondary satellite data, falling back to old data")
            df_primary = goes_downloader_old(year)
    
    if not ignore_quality and secondary_flag:
        try:
            files_secondary = [f"data/downloads/{year}_secondary/{file}" for file in os.listdir(f"data/downloads/{year}_secondary/")]
            goes_ts_secondary = ts.TimeSeries(files_secondary, concatenate=True, allow_errors=True)
            df_secondary = goes_ts_secondary.to_dataframe()
            df_secondary.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
            df_secondary.rename(columns={"xrsb":"xl", "xrsb_quality":"xl_quality"}, inplace=True)
            df_secondary.reset_index(inplace=True)
            df_secondary.rename(columns={"index":"time_tag"}, inplace=True)
        except IndexError:
            print("Index error in secondary satellite data, falling back to primary satellite data")
            secondary_flag = False
            pass

    if not ignore_quality and secondary_flag:
        df_secondary.set_index('time_tag', inplace=True)
        df_primary.set_index('time_tag', inplace=True)
        df_combined = df_primary.join(df_secondary, lsuffix='_primary', rsuffix='_secondary', how='left')
        
        df_combined['xl'] = np.where(df_combined['xl_quality_primary'] > df_combined['xl_quality_secondary'], 
                         df_combined['xl_secondary'], df_combined['xl_primary'])
        df_combined['xl_quality'] = np.where(df_combined['xl_quality_primary'] > df_combined['xl_quality_secondary'], 
                             df_combined['xl_quality_secondary'], df_combined['xl_quality_primary'])
        df_combined['src'] = np.where(df_combined['xl_quality_primary'] > df_combined['xl_quality_secondary'], 
                          'secondary', 'primary')
        
        df_combined.reset_index(inplace=True)
        df_primary = df_combined[['time_tag', 'xl', 'xl_quality', 'src']]

        print(f"Finished processing data for {year}")
        return df_primary
    else:
        df_primary.loc[:, "src"] = "primary"
        print(f"Finished processing data for {year}")
        return df_primary

def goes_downloader_old(year: int) -> pd.DataFrame:
    """
    Function to download legacy GOES data for a given year.

    Parameters:
    year (int): The year for which to download data.

    Returns:
    pd.DataFrame: DataFrame containing the downloaded data.
    """
    satellite_years = {'08': np.arange(1995, 2003, 1), '12': np.arange(2003, 2007, 1),
                        '11': np.arange(2007, 2008, 1), '10': np.arange(2008, 2010, 1),
                        '14': np.arange(2010, 2011, 1), '15': np.arange(2011, 2017, 1),
                        '16': np.arange(2017, 2025, 1)}
    

    satellite = [satellite for satellite, years in satellite_years.items() if year in years][0]

    tstart=f"{year}-01-01 00:00:00"
    tend=f"{year}-12-31 23:59:59"

    result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"), a.goes.SatelliteNumber(satellite))
    print(result)
    files = Fido.fetch(result, path="data/downloads/")

    clear_output()

    goes_ts = ts.TimeSeries(files, concatenate=True)
    df = goes_ts.to_dataframe()

    df.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
    df.rename(columns={"xrsb":"xl", "xrsb_quality":"xl_quality"}, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index":"time_tag"}, inplace=True)
    df.loc[:, "src"] = "primary"
    return df

def legacy_handler_MANUAL(file: str) -> pd.DataFrame:
    """
    Processes legacy GOES data manually downloaded from the GOES archive.

    Parameters:
    file (str): Path to the NetCDF file.

    Returns:
    pd.DataFrame: DataFrame containing the processed data.
    """
    with nc.Dataset(file) as dataset:
        data = {var: dataset.variables[var][:] for var in dataset.variables}
        time_start = dataset.getncattr("time_coverage_start")
        time_end = dataset.getncattr("time_coverage_end")
        df = pd.DataFrame(data)

    df.drop(columns=["xrsa_flux", "xrsa_flags", "xrsa_num", "xrsb_num", "xrsa_flags_excluded", "xrsb_flags_excluded", "time"], inplace=True)
    df.rename(columns={"xrsb_flux": "xl", "xrsb_flags": "xl_quality"}, inplace=True)
    df["time_tag"] = pd.date_range(start=time_start, end=time_end, freq="1min", tz="UTC").tz_localize(None)
    df.loc[:, "src"] = "primary"

    return df

def saturation_fix(df):
    df["time_tag"] = pd.to_datetime(df["time_tag"], format="mixed")
    threshold = 2e-3
    mask = df["xl"] >= threshold
    flux = df["xl"].values
    mask = mask.values
    mask = mask.astype(float)
    mask[mask == 0] = np.nan
    masked_flux = flux * mask

    regions = np.split(masked_flux, np.where(np.diff(mask) != 0)[0]+1)
    regions = [r for r in regions if r[0] >= threshold]
    regions = [r for r in regions if len(r) > 1]

    for region in regions:
        diff = np.diff(region)
        zero_diff = np.where(np.abs(diff) < 0.0000001)[0]
        zero_diff = np.append(zero_diff, zero_diff+1)
        zero_diff = np.append(zero_diff, zero_diff+2)
        zero_diff = np.unique(zero_diff)
        if len(zero_diff) > 2:
            region_sat = region[zero_diff]
            idx = np.where(np.isin(flux, region_sat))
            x = np.arange(0, len(region_sat))
            y = np.copy(region_sat)
            y[len(y)//2] = np.max(region_sat) +0.5*np.max(region_sat)
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            region_sat = p(x)
            region[zero_diff] = region_sat
            flux[idx] = region_sat
            print("Saturation corrected")

        df["xl"] = flux
    return df


def flare_vis(input_date: Union[str, pd.Timestamp], flarelist: pd.DataFrame, v: bool = False) -> None:
    """
    Function to visualize flare events.

    Parameters:
    input_date (Union[str, pd.Timestamp]): The date to visualize.
    flarelist (pd.DataFrame): DataFrame containing flare events.
    v (bool): Verbose flag.

    Returns:
    None
    """
    input_date = pd.to_datetime(input_date)

    # Check if the input date is in the range of the flarelist
    if input_date < flarelist["tstart"].min() or input_date > flarelist["tend"].max():
        raise ValueError("Input date out of range")
    
    # Find the event closest to the input date
    flarelist.loc[:, "tstart"] = pd.to_datetime(flarelist["tstart"], format="mixed")
    flarelist.loc[:, "tpeak"] = pd.to_datetime(flarelist["tpeak"], format="mixed")
    flarelist.loc[:, "tend"] = pd.to_datetime(flarelist["tend"], format="mixed")

    flarelist_delta = np.abs(flarelist["tstart"] - input_date)
    idx = flarelist_delta.idxmin()
    event = flarelist.loc[idx]
    if v:
        print(event)

    # Group by events with the same flare_id
    grouped = flarelist.groupby("flare_id")
    flare_id = event["flare_id"]
    group = grouped.get_group(flare_id)
    if v:
        print(group)

    input_date_year = input_date.year
    df_goes = pd.read_csv(f"data/goes_{input_date_year}.csv")
    df_goes["time_tag"] = pd.to_datetime(df_goes["time_tag"], format="mixed")
    df_goes.set_index("time_tag", inplace=True)

    if len(group) > 1:
        goes_event = df_goes.loc[group["tstart"].min() - pd.Timedelta(minutes=30):group["tend"].max() + pd.Timedelta(minutes=30)]
    else:
        goes_event = df_goes.loc[event["tstart"] - pd.Timedelta(minutes=30):event["tend"] + pd.Timedelta(minutes=30)]

    plt.figure(figsize=(10, 6))
    plt.plot(goes_event["xl"], label="Flux")
    plt.yscale("log")
    plt.ylabel("Flux")
    plt.xlabel("Time")

    for i, row in group.iterrows():
        plt.axvline(row["tstart"], color="red", lw=0.5, ls="--")
        plt.axvline(row["tpeak"], color="green", lw=0.5, ls="--")
        plt.axvline(row["tend"], color="blue", lw=0.5, ls="--")

    if len(group) > 1:
        plt.title(f"Flare {flare_id} // Class: {event['fclass_full']} // {len(group)} events")
        plt.text(0.05, 0.9, "All events: \n" + ", ".join(group["fclass_full"].values), ha='left', transform=plt.gca().transAxes)
        plt.text(0.05, 0.8, "Earliest event: \n" + group["tstart"].min().strftime("%Y-%m-%d %H:%M:%S"), ha='left', transform=plt.gca().transAxes)
    else:
        plt.title(f"Flare {flare_id} // Class: {event['fclass_full']}")
        plt.text(0.05, 0.9, "Closest event: \n" + event["fclass_full"], ha='left', transform=plt.gca().transAxes)

    plt.show()
