# imports
import os
import sys
import pandas as pd
import netCDF4
import numpy as np
import datetime as dt


# filepaths
cmip6Dir = os.path.dirname(os.path.dirname(__file__)) + r"/cmip6/nasa"
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
 

# globals
# -- environment variables
pathway = sys.argv[1]
modelNum = int(sys.argv[2]) - 1
# -- model names
models = os.listdir(cmip6Dir + r"/{}".format(pathway))
model = models[modelNum]
# -- ucrb stuff
noaaMonthly = pd.read_csv(processedDir + r"/NOAA/NOAA_UCRBMonthly.csv")
stations = sorted(set(noaaMonthly["NAME"]))
noaaLatLonElev = {}
for station in stations:
    lat = list(set(noaaMonthly.loc[noaaMonthly["NAME"] == station, "LAT"].values))[0]
    lon = list(set(noaaMonthly.loc[noaaMonthly["NAME"] == station, "LON"].values))[0]
    elev = list(set(noaaMonthly.loc[noaaMonthly["NAME"] == station, "ELEV"].values))[0]
    noaaLatLonElev[station] = {"lat": lat, "lon": lon, "elev": elev}


# function for calculating distances on a spheroid
def CalculateGeographicDistance(point1, point2, convert2Radians=True):
    # define the radius of Earth (km)
    r = 6371.

    # if values need to be converted to radians, do so
    lon1 = np.deg2rad(point1[0]) if convert2Radians else point1[0]
    lat1 = np.deg2rad(point1[1]) if convert2Radians else point1[1]
    lon2 = np.deg2rad(point2[0]) if convert2Radians else point2[0]
    lat2 = np.deg2rad(point2[1]) if convert2Radians else point2[1] 
    elev1 = point1[2] / 1e3
    elev2 = point2[2] / 1e3
    lonDiff = abs(lon2 - lon1)

    # arc length calculation
    num = np.sqrt(np.power(np.cos(lat2)*np.sin(lonDiff), 2.) + np.power(np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lonDiff), 2.))
    dem = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lonDiff)
    arcRad = np.arctan2(num, dem)
    arcLength = r * arcRad

    # distance calculation (not exactly the right formula for adding elevation on a geoid, but close enough here)
    return np.sqrt((arcLength ** 2.) + ((elev2 - elev1) ** 2.))


# find the grid cell closest to each UCRB station
def MatchStationToGrid(nc4Lats, nc4Lons): 
    matchStationToGrid = {}
    for station in stations:
        stationLoc = [noaaLatLonElev[station]["lon"], noaaLatLonElev[station]["lat"], noaaLatLonElev[station]["elev"]]
        nearest = np.Inf
        for y, nc4Lat in enumerate(nc4Lats):
            for x, nc4Lon in enumerate(nc4Lons):
                distBetween = CalculateGeographicDistance(stationLoc, [nc4Lon, nc4Lat, 0.0])
                if distBetween < nearest:
                    nearest = distBetween
                    matchStationToGrid[station] = {"grid": [x, y], "lonlat": [nc4Lons[x], nc4Lats[y]]}
    return matchStationToGrid


# read each file for pathway/model NASA CMIP6 data, and concatenate to a single pathway/model .csv
def ReadCMIP6PathwayModel():
    # a dictionary to hold all of the cmip6 info
    pathwayModelDict = {}
    
    # all of the daily files 
    dailyFiles = os.listdir(cmip6Dir + r"/{}/{}".format(pathway, model))
    dailyFiles = [dailyFile for dailyFile in dailyFiles if ".nc4" in dailyFile]

    # if there aren't any files in that pathway/model combo, nothing needs to be done 
    if (len(dailyFiles) == 0): return 

    # for the files that do exist
    for dailyFile in dailyFiles:
        # file info
        year = int(dailyFile[-8:].split(".")[0]) 
        wvarStr = "PRCP" if "PR" in dailyFile else "TAVG"

        # extract the daily data
        dailyData = netCDF4.Dataset(cmip6Dir + r"/{}/{}/{}".format(pathway, model, dailyFile))
        nc4Times = dailyData.variables["time"][:]
        nc4Lats = dailyData.variables["lat"][:]
        nc4Lons = dailyData.variables["lon"][:]
        nc4Wvar = dailyData.variables["pr"][:] if wvarStr == "PRCP" else dailyData.variables["tas"][:]  # 3d matrix with dims (time, lat, lon)
        
        # nc4 file transformations to physical
        startDate = "{}/01/01".format(year)
        daySteps = nc4Times - nc4Times[0]
        dates = np.array([dt.datetime.strptime(startDate, "%Y/%m/%d") + dt.timedelta(days=dayStep) for dayStep in daySteps])
        lats = nc4Lats
        lons = nc4Lons - 360.
        # -- precipitation in is units of [kg m^-2 s^-1], and we need to convert to [m] --> [kg m^-2 s^-1] * [m^3 kg^-1] * [s], or [prcp]*[1/density]*[length of day] 
        # -- temperature is in units of [K], so we just need to subtract 273.15
        wvar = nc4Wvar * (1./998.) * (24.*60.*60.) if wvarStr == "PRCP" else nc4Wvar - 273.15
        
        # building the grid-to-station dictionary
        cmip6StationDict = MatchStationToGrid(lats, lons)

        # for each station
        for station in stations:
            gridX, gridY = cmip6StationDict[station]["grid"][0], cmip6StationDict[station]["grid"][1]
            for d, date in enumerate(dates):
                monthName = date.strftime("%b") 
                completeKey = (pathway, model, date.year, monthName, date.day, station)
                # figure out which PRCP/TAVG value we should use
                dailyWvar = wvar[d, gridY, gridX]
                # if we haven't seen this key yet, create it, otherwise just append to it
                # -- should ALWAYS be PR first, TAS second because the files are alphabetical
                if completeKey not in pathwayModelDict.keys():
                    pathwayModelDict[completeKey] = [pathway, model, date.year, monthName, date.day, station, dailyWvar]
                else:
                    pathwayModelDict[completeKey].append(dailyWvar)
    
    # create a new dataframe from this dictionary
    df = pd.DataFrame.from_dict(pathwayModelDict, orient="index", columns=["PATHWAY", "MODEL", "YEAR", "MONTH", "DAY", "STATION", "PRCP", "TAVG"])
    return df


# execuate the main file
if __name__ == "__main__":
    # combine all of the CMIP6 data into a single dataframe 
    cmip6PMDF = ReadCMIP6PathwayModel()
    
    # only save a .csv if there's something to save
    if cmip6PMDF is not None:
        cmip6PMDF.to_csv(cmip6Dir + r"/{}/{}/Raw_NASACMIP6_{}_{}_Daily.csv".format(pathway, model, pathway, model), index=False) 
    
