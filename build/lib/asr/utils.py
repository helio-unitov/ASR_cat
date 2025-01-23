from typing import Dict, List
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries as ts
from IPython.display import clear_output
import pandas as pd

def goes_downloader(year: int) -> pd.DataFrame:
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
    print(result)
    
    # Fetch the data files
    files: List[str] = Fido.fetch(result, path="data/downloads/")

    # Clear the output to keep the notebook clean
    clear_output()

    # Load the data into a TimeSeries object and convert to DataFrame
    goes_ts = ts(files, concatenate=True)
    df: pd.DataFrame = goes_ts.to_dataframe()

    # Drop unnecessary columns and rename others
    df.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
    df.rename(columns={"xrsb": "xl", "xrsb_quality": "xl_quality"}, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "time_tag"}, inplace=True)

    return df