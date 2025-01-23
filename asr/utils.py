from typing import Dict, List
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries as ts
import pandas as pd
import matplotlib.pyplot as plt

def goes_downloader(year: int, dpath="data/", v=False) -> pd.DataFrame:
    """
    Downloads GOES satellite data for a given year, processes it, and returns it as a DataFrame.

    Args:
        year (int): The year for which to download the data.

    Returns:
        pd.DataFrame: A DataFrame containing the processed GOES satellite data.

    PSA: This docstring was produced with the aid of AI.
    """

    # Dictionary mapping satellite numbers to their operational years
    satellite_years: Dict[str, np.ndarray] = {
        '08': np.arange(1995, 2003, 1), 
        '12': np.arange(2003, 2007, 1),
        '11': np.arange(2007, 2008, 1), 
        '10': np.arange(2008, 2010, 1),
        '14': np.arange(2010, 2011, 1), 
        '15': np.arange(2011, 2017, 1),
        '16': np.arange(2017, 2025, 1)
    }

    # Determine the satellite number for the given year
    satellite: str = [satellite for satellite, years in satellite_years.items() if year in years][0]

    # Define the start and end times for the data query, as we want to download one full year of data this simply starts at the beginning of the year and ends at the end of the year
    tstart: str = f"{year}-01-01 00:00:00"
    tend: str = f"{year}-12-31 23:59:59"

    # Search for the data using Fido
    result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"), a.goes.SatelliteNumber(satellite))
    if v:
        print(result)
    
    # Fetch the data files
    files: List[str] = Fido.fetch(result, path=f"{dpath}downloads/")

    # Load the data into a TimeSeries object and convert to DataFrame
    goes_ts = ts(files, concatenate=True)
    df: pd.DataFrame = goes_ts.to_dataframe()

    # Drop unnecessary columns and rename others
    df.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
    df.rename(columns={"xrsb": "xl", "xrsb_quality": "xl_quality"}, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "time_tag"}, inplace=True)

    if v:
        print("\n ############################################# \n ############################################# \n")
        print(df)
    return df


def flare_vis(input_date, flarelist, v=False):

    input_date = pd.to_datetime(input_date)

    # check if the input date is in the range of the flarelist
    if input_date < flarelist["tstart"].min() or input_date > flarelist["tend"].max():
        raise ValueError("Input date out of range")
    # find the event closest to the input date
    flarelist["tstart"] = pd.to_datetime(flarelist["tstart"], format="mixed")
    flarelist["tpeak"] = pd.to_datetime(flarelist["tpeak"], format="mixed")
    flarelist["tend"] = pd.to_datetime(flarelist["tend"], format="mixed")

    flarelist_delta = np.abs(flarelist["tstart"] - input_date)
    idx = flarelist_delta.idxmin()
    event = flarelist.loc[idx]
    if v:
        print(event)

    # group by events with the same flare_id
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
        # write in plt.text the class of the other events and the requested one only in bold
        plt.text(0.05, 0.9, "All events: \n" + ", ".join(group["fclass_full"].values), ha='left', transform=plt.gca().transAxes)
        plt.text(0.05, 0.8, "Earliest event: \n" + group["tstart"].min().strftime("%Y-%m-%d %H:%M:%S"), ha='left', transform=plt.gca().transAxes)
    else:
        plt.title(f"Flare {flare_id} // Class: {event['fclass_full']}")
        plt.text(0.05, 0.9, "Closest event: \n" + event["fclass_full"], ha='left', transform=plt.gca().transAxes)

    plt.show()
