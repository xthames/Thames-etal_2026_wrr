# imports
import os
import sys
import pandas as pd
import time
import subprocess
import netCDF4
import numpy as np
import datetime as dt


# filepaths
statecuDir = os.path.dirname(os.path.dirname(__file__))
cmip6Dir = statecuDir + r"/cmip6/hydroclimate"
processedDir = statecuDir + r"/processed"


# reduce the complete .csv file with all the monthly links to just links with prcp, tmin, tmax
def ReduceLinks():
    # read in the complete .csv
    linksDF = pd.read_csv(cmip6Dir + r"/CMIP6Hydroclimate_Gridded_MonthlyData_DownloadLinks_Complete.csv")
    
    # divide into three sets
    # -- (1) observations
    indices2Drop = [True if "control" not in el else False for el in linksDF["Simulation Name"].values]
    obsLinksDF = linksDF.drop(linksDF[indices2Drop].index)
    # -- (2) prcp/tmin/tmax
    indices2Drop = [True if (linksDF.loc[i, "Variables ID"] not in ["prcp", "tmin", "tmax"] or "control" in linksDF.loc[i, "Simulation Name"]) else False for i in range(len(linksDF))]
    ptLinksDF = linksDF.drop(linksDF[indices2Drop].index)
    # -- (3) runoff
    indices2Drop = [True if ("runoff" not in linksDF.loc[i, "Variables ID"] or "control" in linksDF.loc[i, "Simulation Name"]) else False for i in range(len(linksDF))]
    runoffLinksDF = linksDF.drop(linksDF[indices2Drop].index)
    
    # create these .csvs
    obsLinksDF.to_csv(cmip6Dir + r"/CMIP6Hydroclimate_Gridded_MonthlyData_DownloadLinks_OBS.csv", index=False)
    ptLinksDF.to_csv(cmip6Dir + r"/CMIP6Hydroclimate_Gridded_MonthlyData_DownloadLinks_PT.csv", index=False) 
    runoffLinksDF.to_csv(cmip6Dir + r"/CMIP6Hydroclimate_Gridded_MonthlyData_DownloadLinks_RUNOFF.csv", index=False)


# make all the necessary directories if they haven't yet been made
def MakeDirectories():
    pathways = ["observations", "historical", "ssp585"]
    for pathway in pathways:
        models = ["Daymet", "Livneh"] if pathway == "observations" else ["ACCESS-CM2", "BCC-CSM2-MR", "CNRM-ESM2-1", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-MM"]
        for model in models:
            os.system("mkdir -p {}/{}/{}/".format(cmip6Dir, pathway, model))


# get the observations, perform the station alignment simplifications so that we're not storing ~1GB files
def wgetCMIP6():  
    # find the download .csv files in the hydroclimate folder
    hydroclimateDirList = os.listdir(cmip6Dir)  
    downloadCSVs = [f for f in hydroclimateDirList if os.path.isfile(cmip6Dir + "/" + f) and "Complete" not in f]
    downloadCSVs = [f for f in downloadCSVs if "RUNOFF" not in f]
    for downloadFile in downloadCSVs: 
        df = pd.read_csv(cmip6Dir + "/" + downloadFile)
        for i in range(len(df)):
            # get the entry 
            obsEntry = df.loc[i]
            wvar = obsEntry["Variables ID"].upper()
            forcing = obsEntry["Ref. Met. Focring"]
            period = obsEntry["Period"]
            # setting up the filepath
            if obsEntry["Climate Model"] != "-":
                if period == "1980_2019":
                    sortDir = "historical"
                else:
                    sortDir = "ssp585"
                downscalingMethod = obsEntry["Downscaling Method"]
                climateModel = obsEntry["Climate Model"]
                fPath = cmip6Dir + r"/{}/{}".format(sortDir, climateModel)
                fName = "Monthly{}_{}{}_{}.nc".format(wvar, forcing, downscalingMethod, period)
            else:
                sortDir = "observations"
                fPath = cmip6Dir + r"/{}/{}".format(sortDir, forcing)
                fName = "Monthly{}_{}.nc".format(wvar, period)
            # wget the file
            wgetStr = "wget --retry-connrefused --tries=100 --waitretry=45 --no-check-certificate -a {}/output.log -O {}/{}".format(fPath, fPath, fName)
            wgetCmdStr = wgetStr.split(" ")
            url = obsEntry["Download Links"]
            subprocess.call([*wgetCmdStr, url])
            
            # extract the station data from the CONUS files, save, and delete the full set
            ExtractStationData(fPath, fName, wvar.lower())


# turn the daily observations into monthly observations .csv aligned with our stations
def ExtractStationData(filePath, fileName, clivar): 
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
    
    # extract the monthly data
    dailyData = netCDF4.Dataset("{}/{}".format(filePath, fileName))
    nc4Times = dailyData.variables["time"][:]
    nc4Lats = dailyData.variables["lat"][:]
    nc4Lons = dailyData.variables["lon"][:]
    
    # for the climate variable, need to readjust when reading for runoff
    clivar = "runoff" if "runoff" in clivar else clivar
    nc4Wvar = dailyData.variables[clivar][:]

    # nc4 file transformations to physical
    monthSteps = nc4Times - nc4Times[0]
    startYear = int(fileName[-12:-8])
    startDate, startDT, daySteps = "{}/01/01".format(startYear), dt.date(startYear, 1, 1), [] 
    for monthStep in monthSteps:
        addYear = int(monthStep/12)
        addMonth = int(monthStep - 12*addYear)
        daySteps.append((dt.date(startYear+addYear, 1+addMonth, 1) - startDT).days)
    dates = np.array([dt.datetime.strptime(startDate, "%Y/%m/%d") + dt.timedelta(days=dayStep) for dayStep in daySteps])
    # -- precipitation in is units of [mm/month], and we need to convert to [m] --> [mm] / 1000 
    # -- temperature is in units of [degC], so no change
    wvar = nc4Wvar * (1./1000.) if clivar in ["prcp", "runoff"] else nc4Wvar 
    lats, lons = nc4Lats, nc4Lons

    # building the grid-to-station dictionary
    cmip6StationDict = MatchStationToGrid(lats, lons) 

    # for each station
    nc4Dict = {}
    for station in stations:
        gridX, gridY = cmip6StationDict[station]["grid"][0], cmip6StationDict[station]["grid"][1]
        for d, date in enumerate(dates):
            monthName = date.strftime("%b") 
            nc4Key = (date.year, monthName, station)
            # figure out which PRCP/TAVG value we should use
            monthlyWvar = wvar[d, gridY, gridX]
            nc4Dict[nc4Key] = [date.year, monthName, station, monthlyWvar]
    
    # create a new dataframe from this dictionary
    df = pd.DataFrame.from_dict(nc4Dict, orient="index", columns=["YEAR", "MONTH", "STATION", clivar.upper()])
    df.to_csv("{}/{}.csv".format(filePath, fileName[:-3]), index=False)
    del dailyData, nc4Times, nc4Lats, nc4Lons, df

    # remove the huge CONUS file now that we've extracted the stations
    os.remove("{}/{}".format(filePath, fileName))


# check each pathway, model directory to see if all of the files we expect to be there are there AND POPULATED
def CheckDirectories():
    missingDict = {}
    for pathway in pathways:
        for model in models: 
            # skip the ones that don't exist
            if (pathway == "ssp370" and model in ["CESM2-WACCM", "GFDL-CM4", "GFDL-CM4_gr2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "KIOST-ESM", "NESM3"]) or \
               (pathway == "ssp126" and model in ["CESM2-WACCM", "GFDL-CM4", "GFDL-CM4_gr2"]) or \
               (pathway == "ssp245" and model in ["HadGEM3-GC31-MM"]):
                continue
            
            # path to the directory
            dp = cmip6Dir + "/{}/{}".format(pathway, model)
            # get the number of pr, tas files in this directory of any size
            prFiles = [f for f in os.listdir(dp) if "PR" in f]
            tasFiles = [f for f in os.listdir(dp) if "TAS" in f]
            prSuccesses, tasSuccesses = 0, 0 

            # for each file, check if it downloaded successfully
            for prFile in prFiles:
                if os.path.isfile(dp + "/{}".format(prFile)) and os.path.getsize(dp + "/{}".format(prFile)) > 1:
                    prSuccesses += 1
                else:
                    year = prFile[-8:].split(".")[0]
                    missingDict[(pathway, model, "PR", year)] = True
            for tasFile in tasFiles:
                if os.path.isfile(dp + "/{}".format(tasFile)) and os.path.getsize(dp + "/{}".format(tasFile)) > 1:
                    tasSuccesses += 1
                else:
                    year = tasFile[-8:].split(".")[0]
                    missingDict[(pathway, model, "TAS", year)] = True
            # percentage of how many of the files were downloaded -- SHOULD BE 100%
            prSuccesses = prSuccesses / len(prFiles) * 100
            tasSuccesses = tasSuccesses / len(tasFiles) * 100
    
    # a more condensed printout of non-downloaded files
    allDownloaded = True
    for k, isMissing in missingDict.items():
        p, m, w, y = k[0], k[1], k[2], k[3]
        if isMissing:
            allDownloaded = False
            print("{} {} {} {}".format(p.upper(), m.upper(), w.upper(), y.upper()))
    if allDownloaded:
        print("** ALL CMIP6 FILES SUCCESSFULLY DOWNLOADED **")
    else:
        print("** CMIP6 FILES PRINTED ABOVE ARE MISSING FROM THE DOWNLOAD **")
        print("** SUBMITTING THE JOB AGAIN SHOULD RECOVER MISSING FILES **")


# when the script is actually called
if __name__ == "__main__":
    # read in what we want to do
    task = sys.argv[1].lower() if len(sys.argv) > 1 else "na"
    match task:
        case "reducelinks":
            ReduceLinks()
        
        case "makedirs":
            MakeDirectories()
        
        case "wget":    
            # reading in the NOAA data
            noaaMonthly = pd.read_csv(processedDir + r"/NOAA_UCRBMonthly.csv")
            stations = sorted(set(noaaMonthly["NAME"]))
            noaaLatLonElev = {}
            for station in stations:
                lat = list(set(noaaMonthly.loc[noaaMonthly["NAME"] == station, "LAT"].values))[0]
                lon = list(set(noaaMonthly.loc[noaaMonthly["NAME"] == station, "LON"].values))[0]
                elev = list(set(noaaMonthly.loc[noaaMonthly["NAME"] == station, "ELEV"].values))[0]
                noaaLatLonElev[station] = {"lat": lat, "lon": lon, "elev": elev}
            
            # download
            wgetCMIP6()
        
        case "checkdirs":
            CheckDirectories()
        
        case _:
            raise NotImplementedError("Please pass a valid environment variable!")

