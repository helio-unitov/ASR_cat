from typing import Dict, List
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries as ts
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

def goes_downloader_main(year, custom_sat_data=None, ignore_quality: bool = False):

    secondary_flag = None

    if custom_sat_data is not None:
        satellite_years = pd.read_csv(custom_sat_data)
        satellite_years['start_datetime'] = pd.to_datetime(satellite_years['start_datetime'])
        satellite_years = satellite_years[satellite_years['start_datetime'].dt.year == year]
        satellite_years.reset_index(drop=True, inplace=True)
        # download the data
        for i in range(len(satellite_years)):
            tstart = satellite_years['start_datetime'][i]
            tend = satellite_years['end_datetime'][i]
            result_primary = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"), a.goes.SatelliteNumber(int(satellite_years['primary'][i])))
            if satellite_years["secondary"][i] != np.nan and not ignore_quality:
                result_secondary = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"), a.goes.SatelliteNumber(int(satellite_years['secondary'][i])))
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
        return df

    clear_output()

    if year < 2002:
        files_primary = [f"data/downloads/{year}_primary/{file}" for file in os.listdir(f"data/downloads/{year}_primary/")]
        df = legacy_handler(files_primary)
        return df

    files_primary = [f"data/downloads/{year}_primary/{file}" for file in os.listdir(f"data/downloads/{year}_primary/")]
    goes_ts_primary = ts.TimeSeries(files_primary, concatenate=True)
    df_primary = goes_ts_primary.to_dataframe()
    df_primary.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
    df_primary.rename(columns={"xrsb":"xl", "xrsb_quality":"xl_quality"}, inplace=True)
    df_primary.reset_index(inplace=True)
    df_primary.rename(columns={"index":"time_tag"}, inplace=True)
    
    if not ignore_quality and secondary_flag:
        files_secondary = [f"data/downloads/{year}_secondary/{file}" for file in os.listdir(f"data/downloads/{year}_secondary/")]
        goes_ts_secondary = ts.TimeSeries(files_secondary, concatenate=True)
        df_secondary = goes_ts_secondary.to_dataframe()
        df_secondary.drop(columns=["xrsa", "xrsa_quality"], inplace=True)
        df_secondary.rename(columns={"xrsb":"xl", "xrsb_quality":"xl_quality"}, inplace=True)
        df_secondary.reset_index(inplace=True)
        df_secondary.rename(columns={"index":"time_tag"}, inplace=True)

    if not ignore_quality and secondary_flag:

        src = []

        for i in tqdm(range(len(df_primary)), desc="Checking quality"):
            # check at the same date and time if the quality of the primary satellite is worse than the secondary satellite, if so, replace the primary satellite data with the secondary satellite data
            # the same date in the secondary satellite data may not be at the same index as the primary satellite data. 
            # Therefore, we need to find the index of the secondary satellite data that corresponds to the same date and time as the primary satellite data
            index = df_secondary[df_secondary['time_tag'] == df_primary['time_tag'][i]].index
            if len(index) > 0:
                index = index[0]
                if df_primary['xl_quality'][i] > df_secondary['xl_quality'][index]:
                    df_primary.loc[i, 'xl'] = df_secondary['xl'][index]
                    df_primary.loc[i, 'xl_quality'] = df_secondary['xl_quality'][index]
                    src.append('secondary')
                else:
                    src.append('primary')
            else:
                src.append('primary')
        df_primary.loc[:, 'src'] = src
        return df_primary
    else:
        return df_primary



def goes_downloader_old(year):

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
    # rename the index time_tag
    df.rename(columns={"index":"time_tag"}, inplace=True)
    return df

def legacy_handler(files):
    file = files[0]
    data = fits.getdata(file)
    if len(data.shape) == 2:
        data_hdr = fits.getheader(file)
        data = data.T
        time = pd.date_range(start=data_hdr["DATE-OBS"], periods=data.shape[0], freq='3s')

        data_dict = {"time_tag":time, "xl":np.array(data[:,1]).astype(float)}
        df = pd.DataFrame(data_dict)

        for file in files[1:]:

            data = fits.getdata(file)
            data = data.T
            data_hdr = fits.getheader(file)
            time = pd.date_range(start=data_hdr["DATE-OBS"], periods=data.shape[0], freq='3s')
            data_dict = {"time_tag": time, "xl": np.array(data[:,1]).astype(float)}
            df = pd.concat([df, pd.DataFrame(data_dict)], ignore_index=True)

        # resample the data to 1 minute
        df.set_index("time_tag", inplace=True)
        df = df.resample("1T").mean()
        df.reset_index(inplace=True)
        df.loc[:, "xl_quality"] = 0
        df.loc[:, "src"] = "legacy"

    else:
        data_src = fits.open(files[0])
        data_hdr = data_src[0].header
        data = data_src[2].data
        date = file.split("/")[-1].split(".")[0][4:][:4]+"-"+file.split("/")[-1].split(".")[0][4:][4:6]+"-"+file.split("/")[-1].split(".")[0][4:][6:]
        time = pd.date_range(start=date+" 00:00:00", end=date+" 23:59:59", freq='3s')
        data_dict = {"time_tag":time, "xl":np.array(data[0][1][:,1]).astype(float)}
        df = pd.DataFrame(data_dict)

        for file in files[1:]:
            data_src = fits.open(file)
            data = data_src[2].data
            data_hdr = data_src[0].header
            date = file.split("/")[-1].split(".")[0][4:][:4]+"-"+file.split("/")[-1].split(".")[0][4:][4:6]+"-"+file.split("/")[-1].split(".")[0][4:][6:]
            time = pd.date_range(start=date+" 00:00:00", end=date+" 23:59:59", freq='3s')
            # check if time and np.array(data[0][1][:,1]).astype(float) have the same length, if not, skip the file
            if len(time) != len(np.array(data[0][1][:,1]).astype(float)):
                continue
            data_dict = {"time_tag": time, "xl": np.array(data[0][1][:,1]).astype(float)}
            df = pd.concat([df, pd.DataFrame(data_dict)], ignore_index=True)
        
        # resample the data to 1 minute
        df.sort_values("time_tag", inplace=True)
        df.set_index("time_tag", inplace=True)
        df = df.resample("1T").mean()
        df.reset_index(inplace=True)
        df.loc[:, "xl_quality"] = 0
        df.loc[:, "src"] = "legacy"
        
    return df

def flare_vis(input_date, flarelist, v=False):

    input_date = pd.to_datetime(input_date)

    # check if the input date is in the range of the flarelist
    if input_date < flarelist["tstart"].min() or input_date > flarelist["tend"].max():
        raise ValueError("Input date out of range")
    # find the event closest to the input date
    flarelist.loc[:, "tstart"] = pd.to_datetime(flarelist["tstart"], format="mixed")
    flarelist.loc[:, "tpeak"] = pd.to_datetime(flarelist["tpeak"], format="mixed")
    flarelist.loc[:, "tend"] = pd.to_datetime(flarelist["tend"], format="mixed")

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
