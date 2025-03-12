import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
from astropy.io import fits
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
from sitools2 import SdoClientMedoc
import trackpy as tp

sdo = SdoClientMedoc()

def medoc_query(start_date: datetime, peak_date: datetime, end_peak_date: datetime, end_date: datetime, wave: int = 171) -> None:
    """Query MEDOC for SDO data."""
    sdo_data_start = sdo.search(DATES=[start_date, end_date], NB_RES_MAX=1, waves=[wave])
    sdo_data_peak = sdo.search(DATES=[peak_date, end_peak_date], NB_RES_MAX=1, waves=[wave])
    sdo_data_list = sdo_data_start + sdo_data_peak
    for data in sdo_data_list:
        data.get_file(target_dir="data/results_dir/", segment=['image_lev1'], DECOMPRESS=True)

def jsoc_query(start_date: np.datetime64, peak_date: np.datetime64, end_peak_date: np.datetime64, end_date: np.datetime64, wave: int = 171, email: str = "michele.berretti@roma2.infn.it", dpath: str = "data/results_dir/") -> None:
    """Query JSOC for AIA data."""
    query_aia = Fido.search(
        a.Time(start_date, end_date) | 
        a.Time(peak_date, end_peak_date),
        a.jsoc.Series.aia_lev1,
        a.jsoc.Notify(email),
        a.jsoc.Segment.image_lev1,
        a.jsoc.Keyword("WAVELNTH") == wave
    )
    res = query_aia['jsoc']
    aia_files = [res[0], res[1]]
    for aia_file in aia_files: 
        dl = Fido.fetch(aia_file, path=dpath)
        # retry if download failed
        dl = Fido.fetch(dl, path=dpath)

def get_flares_location(datapath: str, dpath: str, flarelist: pd.DataFrame, source: str, start_clean: bool = False) -> None:
    """Get the location of flares."""
    flarelist["tstart"] = pd.to_datetime(flarelist["tstart"], format="mixed")
    flarelist = flarelist[flarelist["tstart"] > np.datetime64('2010-05-13')]
    tstart = np.array(flarelist["tstart"])
    peak = np.array(flarelist["tpeak"])
    flare_x = -1 * np.ones(len(flarelist))
    flare_y = -1 * np.ones(len(flarelist))

    if start_clean:
        try:
            os.remove(datapath + "flare_x.npy")
            os.remove(datapath + "flare_y.npy")
        except FileNotFoundError:
            pass

    last_index = 0
    try:
        print("Loading previous data")
        flare_x = np.load(datapath + "flare_x.npy")
        flare_y = np.load(datapath + "flare_y.npy")
        last_index = np.where(flare_x == -1)[0][0]
    except FileNotFoundError:
        print("No previous data found")

    for j in range(last_index, len(flarelist)):
        try:
            filelist = [f for f in os.listdir("data/results_dir/")]
            for f in filelist:
                os.remove(os.path.join("data/results_dir/" + f))

            if source == "medoc":
                start_time = pd.to_datetime(tstart[j])
                start_date = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, 0)
                end_time = start_time + np.timedelta64(60, 's')
                end_date = datetime(end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, 0)
                peak_time = pd.to_datetime(peak[j])
                peak_date = datetime(peak_time.year, peak_time.month, peak_time.day, peak_time.hour, peak_time.minute, 0)
                end_peak_time = peak_time + np.timedelta64(60, 's')
                end_peak_date = datetime(end_peak_time.year, end_peak_time.month, end_peak_time.day, end_peak_time.hour, end_peak_time.minute, 0)
                medoc_query(start_date, peak_date, end_peak_date, end_date, wave=171)
            elif source == "jsoc":
                start_time = np.datetime64(tstart[j])
                end_time = start_time + np.timedelta64(10, 's')
                peak_time = np.datetime64(peak[j])
                end_peak_time = peak_time + np.timedelta64(10, 's')
                jsoc_query(start_time, peak_time, end_peak_time, end_time, wave=171)
            else:
                raise ValueError("Source not recognized")

            downloaded = sorted(glob.glob("data/results_dir/*.fits"))
            data = Map(downloaded, sequence=True)
            coaligned = data  # mc_coalign(data)
            diffimg = coaligned[1].data - coaligned[0].data
            f = tp.locate(diffimg, 11, invert=False)
            g = f.iloc[f['mass'].idxmax()]
            x = g.iloc[1]
            y = g.iloc[0]

            print("Flare number: ", j + 1, " / ", len(flarelist))
            flare_x[j] = x
            flare_y[j] = y
            np.save(datapath + "flare_x.npy", flare_x)
            np.save(datapath + "flare_y.npy", flare_y)
            flarelist["flare_x"] = flare_x
            flarelist["flare_y"] = flare_y
            flarelist.to_csv(datapath + f"flarelist_positions.csv")

        except Exception as e:
            print(f"Error processing flare {j}: {e}")
            flare_x[j] = np.nan
            flare_y[j] = np.nan
            np.save(datapath + "flare_x.npy", flare_x)
            np.save(datapath + "flare_y.npy", flare_y)
            flarelist["flare_x"] = flare_x
            flarelist["flare_y"] = flare_y
            flarelist.to_csv(datapath + f"flarelist_positions.csv")


def coordinate_change(source, start_clean=False):

    if start_clean:
        lon = np.ones(len(source))*-9999
        lat = np.ones(len(source))*-9999
        hgs_lon = np.ones(len(source))*-9999
        hgs_lat = np.ones(len(source))*-9999
        start=0
    elif not start_clean:
        try:
            lon = np.load("flare_positions.npy")[0]
            lat = np.load("flare_positions.npy")[1]
            hgs_lon = np.load("flare_positions.npy")[2]
            hgs_lat = np.load("flare_positions.npy")[3]
            # start should be the last non--9999 value
            start = np.where(lon==-9999)[0][0]
        except:
            lon = np.ones(len(source))*-9999
            lat = np.ones(len(source))*-9999
            hgs_lat = np.ones(len(source))*-9999
            hgs_lon = np.ones(len(source))*-9999
            start=0
            print("Error loading previous positions, starting from scratch")
    


    for j in range(start, len(source)):
        clear_output()
        print(f"Processing {j+1}/{len(source)}")
        try:
            start_time = pd.to_datetime(tstart[j])
            start_date = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, 0)
            end_time = start_time + np.timedelta64(60, 's')
            end_date = datetime(end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, 0)
            os.system("rm -rf data/results_dir/*")
            medoc_query(start_date, end_date, wave=171)
            file = glob.glob("data/results_dir/*.fits")[0]
            aiamap = sunpy.map.Map(file)

            flare_x = source["flare_x"][j]*u.pixel
            flare_y = source["flare_y"][j]*u.pixel
            coord = aiamap.wcs.pixel_to_world(flare_x, flare_y)
            lon[j] = coord.Tx.value
            lat[j] = coord.Ty.value

            coord_hgs = coord.transform_to(frames.HeliographicStonyhurst)
            hgs_lon[j] = coord_hgs.lon.value
            hgs_lat[j] = coord_hgs.lat.value
            np.save("flare_positions.npy", [lon, lat, hgs_lon, hgs_lat])
        except ValueError:
            print("ValueError")
            lat[j] = np.nan
            lon[j] = np.nan
            hgs_lat[j] = np.nan
            hgs_lon[j] = np.nan
            np.save("flare_positions.npy", [lon, lat, hgs_lon, hgs_lat])
            continue
        except IndexError:
            print("IndexError")
            lat[j] = np.nan
            lon[j] = np.nan
            hgs_lat[j] = np.nan
            hgs_lon[j] = np.nan
            np.save("flare_positions.npy", [lon, lat, hgs_lon, hgs_lat])
            continue
        except:
            print("Other error")
            lat[j] = np.nan
            lon[j] = np.nan
            hgs_lat[j] = np.nan
            hgs_lon[j] = np.nan
            np.save("flare_positions.npy", [lon, lat, hgs_lon, hgs_lat])
            continue

    source["Lon [Helioprojective]"] = lon
    source["Lat [Helioprojective]"] = lat

    source["Lon [Heliographic]"] = hgs_lon
    source["Lat [Heliographic]"] = hgs_lat

    return source

