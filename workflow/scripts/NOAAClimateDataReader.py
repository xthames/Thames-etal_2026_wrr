# import
import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import datetime as dt


# filepaths
noaaDir = os.path.dirname(os.path.dirname(__file__)) + r"/noaa"
plotsDir = os.path.dirname(os.path.dirname(__file__)) + r"/plots"
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed/NOAA"


# grabbing the station from the environment argument
stationFile = sys.argv[1]
stationFromFile = stationFile[:stationFile.find("_")]
station = re.sub(r"(\w)([A-Z])", r"\1 \2", stationFromFile)


# dictionary of the primary stations, months and their numbers
primaryStationDict = {"Altenbern": "USC00050214", "Collbran": "USC00051741",
                      "Eagle County": "USW00023063", "Fruita": "USC00053146",
                      "Glenwood Springs": "USC00053359", "Grand Junction": "USC00053489",
                      "Grand Lake": "USC00053500", "Green Mt Dam": "USC00053592",
                      "Kremmling": "USC00054664", "Meredith": "USC00055507",
                      "Rifle": "USC00057031", "Yampa": "USC00059265"}
monthNumDict = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
iMonthNumDict = {v: k for k, v in monthNumDict.items()}


# aggregate the raw daily data into monthly data
def CreateHistoricRawMonthly():
    # dataframe has the following columns:
    # -- Station Name | Station ID | Latitude | Longitude | Elevation | Year | Month | Day | PRCP | TMIN | TMAX
    dailyDFColumns = ["NAME", "ID", "LAT", "LON", "ELEV", "YEAR", "MONTH", "DAY", "PRCP", "TMIN", "TMAX"] 
    
    # read in the daily data
    readDF = pd.read_csv(noaaDir + "/{}".format(stationFile))

    # change the PRCP column from [m] to [mm]
    readDF["PRCP"] = readDF["PRCP"].apply(lambda x: x / 1000)

    # format year/month/day
    splitDate = readDF["DATE"].str.split("-", expand=True)
    years, months, days = splitDate[0].values.astype(int), splitDate[1], splitDate[2].values
    for monthNum in iMonthNumDict:
        months = months.str.replace(monthNum, iMonthNumDict[monthNum])
    readDF["YEAR"], readDF["MONTH"], readDF["DAY"] = years, months, days

    # build a temp dataframe with matching columns as the main one
    dailyDF = pd.DataFrame(columns=dailyDFColumns)
    dailyDF["NAME"] = [station] * readDF.shape[0]
    dailyDF["YEAR"], dailyDF["MONTH"], dailyDF["DAY"] = years, months, days
    dailyDF[["ID", "LAT", "LON", "ELEV", "PRCP", "TMIN", "TMAX"]] = readDF[["STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "PRCP", "TMIN", "TMAX"]]
    del readDF

    # expanded dataframe
    monthlyDFColumns = ["NAME", "ID", "LAT", "LON", "ELEV", "YEAR", "MONTH", "PRCP", "TAVG", "TMIN", "TMAX"]
    monthlyDF = pd.DataFrame(columns=monthlyDFColumns, dtype=object)

    # index for the station
    IDs = set(dailyDF["ID"])
    # for each substation in the grouping...
    for ID in IDs:
        idIndex = dailyDF["ID"] == ID
        years = sorted(set(dailyDF.loc[idIndex, "YEAR"]))
        months = sorted(set(dailyDF.loc[idIndex, "MONTH"]), key=lambda x: dt.datetime.strptime(x, "%b"))

        # make the temporary df
        tempDF = pd.DataFrame(columns=monthlyDFColumns)
        nMonths, nYears = len(months), len(years)
        n = nMonths * nYears

        # metadata for name, ID, etc
        tempDF["NAME"] = [station] * n
        tempDF["ID"] = [ID] * n
        tempDF["LAT"] = list(set(dailyDF.loc[idIndex, "LAT"])) * n
        tempDF["LON"] = list(set(dailyDF.loc[idIndex, "LON"])) * n
        tempDF["ELEV"] = list(set(dailyDF.loc[idIndex, "ELEV"])) * n
        tempDF["YEAR"] = np.repeat(years, nMonths)
        tempDF["MONTH"] = months * nYears

        # NaN arrays for the data to aggregate
        precip = np.full(shape=n, fill_value=np.NaN)
        avgT = np.full(shape=n, fill_value=np.NaN)
        minT = np.full(shape=n, fill_value=np.NaN)
        maxT = np.full(shape=n, fill_value=np.NaN)

        # for each year, month available in the substation...
        for i, year in enumerate(years):
            yearIndex = dailyDF["YEAR"] == year
            for j, month in enumerate(months):
                monthIndex = dailyDF["MONTH"] == month

                # sum the precipitation values
                precipDays = dailyDF.loc[idIndex & yearIndex & monthIndex, "PRCP"].values.astype(float)
                if len(precipDays) and not all(np.isnan(precipDays)):
                    precip[i * nMonths + j] = np.nansum(precipDays[precipDays < 1])

                # average the temperature values
                tempDays = np.ravel(dailyDF.loc[idIndex & yearIndex & monthIndex, ["TMIN", "TMAX"]].values.astype(float))
                if len(tempDays) and not all(np.isnan(tempDays)):
                    avgT[i * nMonths + j] = np.nanmean(tempDays)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        minT[i * nMonths + j] = np.nanmean(tempDays[0::2])
                        maxT[i * nMonths + j] = np.nanmean(tempDays[1::2])

        # add these precip/temp vectors to the dataframe
        tempDF["PRCP"] = precip
        tempDF["TAVG"] = avgT
        tempDF["TMIN"] = minT
        tempDF["TMAX"] = maxT

        # concatenate this into the agreementDF
        monthlyDF = pd.concat([monthlyDF if not monthlyDF.empty else None, tempDF], ignore_index=True)
        del tempDF

    # save this dataframe so that we don't have to remake it
    monthlyDF.to_csv(processedDir + r"/RawNOAA_Monthly_{}.csv".format(stationFromFile), index=False) 
    
    # return the full dataframe
    return monthlyDF


# create the historic monthly biases between primary and secondary stations
def CreateHistoricMonthlyBias(rawMonthlyNOAA):
    # list of the months in the year
    months = sorted(set(rawMonthlyNOAA["MONTH"]), key=lambda x: dt.datetime.strptime(x, "%b"))

    # establish the bias dataframe to return
    biasColumns = ["Primary", "Secondary", "Month", "PRCP Scaling", "PRCP Primary Offset", "PRCP Secondary Offset",
                   "TAVG Scaling", "TAVG Primary Offset", "TAVG Secondary Offset"]
    biasDF = pd.DataFrame(columns=biasColumns, dtype=object)

    # station IDs
    stationIDs = set(rawMonthlyNOAA["ID"].values)

    # get the primary station, secondary stations, using full set of months across all years
    primaryID = primaryStationDict[station]
    primaryIndex = rawMonthlyNOAA["ID"] == primaryID
    secondaryIDs = [ID for ID in stationIDs if ID != primaryID]
    for month in months:
        monthIndex = rawMonthlyNOAA["MONTH"] == month

        # find the bias by investigating primary mean, std --> scale secondaries to that
        primaryPRCPMean = rawMonthlyNOAA.loc[monthIndex & primaryIndex, "PRCP"].mean()
        primaryPRCPOffset = primaryPRCPMean if not np.isnan(primaryPRCPMean) else 0
        primaryTAVGMean = rawMonthlyNOAA.loc[monthIndex & primaryIndex, "TAVG"].mean()
        primaryTAVGOffset = primaryTAVGMean if not np.isnan(primaryTAVGMean) else 0
        primaryPRCPStd = rawMonthlyNOAA.loc[monthIndex & primaryIndex, "PRCP"].std()
        primaryTAVGStd = rawMonthlyNOAA.loc[monthIndex & primaryIndex, "TAVG"].std()
        
        # temporary dataframe
        internalBiasDF = pd.DataFrame(columns=biasColumns)

        # looping over each secondaryID
        for secondaryID in secondaryIDs:
            secondaryIndex = rawMonthlyNOAA["ID"] == secondaryID
            # -- secondary mean, std
            secondaryPRCPMean = rawMonthlyNOAA.loc[monthIndex & secondaryIndex, "PRCP"].mean()
            secondaryPRCPStd = rawMonthlyNOAA.loc[monthIndex & secondaryIndex, "PRCP"].std()
            secondaryTAVGMean = rawMonthlyNOAA.loc[monthIndex & secondaryIndex, "TAVG"].mean()
            secondaryTAVGStd = rawMonthlyNOAA.loc[monthIndex & secondaryIndex, "TAVG"].std()

            # filling the bias DF
            prcpScale = primaryPRCPStd / secondaryPRCPStd if not np.isnan(primaryPRCPStd / secondaryPRCPStd) else 1
            tempScale = primaryTAVGStd / secondaryTAVGStd if not np.isnan(primaryTAVGStd / secondaryTAVGStd) else 1
            secondaryPRCPOffset = secondaryPRCPMean if not np.isnan(secondaryPRCPMean) else primaryPRCPOffset
            secondaryTAVGOffset = secondaryTAVGMean if not np.isnan(secondaryTAVGMean) else primaryTAVGOffset
            # store the biases
            internalBiasDF.loc[len(internalBiasDF)] = [primaryID, secondaryID, month,
                                                       prcpScale, primaryPRCPOffset, secondaryPRCPOffset,
                                                       tempScale, primaryTAVGOffset, secondaryTAVGOffset]

        # add biases to the main dataframe
        biasDF = pd.concat([biasDF if not biasDF.empty else None, internalBiasDF], ignore_index=True)

    # save the biasDF to a dataframe
    biasDF.to_csv(processedDir + r"/RawNOAA_Biases_{}.csv".format(stationFromFile), index=False)

    # return the bias dataframe
    return biasDF


# create the bias-corrected, filled datasets for monthly data and daily data (<- specifically TMIN for synthetic .fd file )
def CreateHistoricProcessedUCRBDatasets(rawMonthlyDF, biasDF):
    # creating the bias-corrected, filled daily data for the eventual .fd file 
    CreateHistoricDailyMinT(pd.read_csv(noaaDir + "/{}".format(stationFile)), biasDF)

    # the columns we'll use for the output .csv
    dfColumns = ["NAME", "LAT", "LON", "ELEV", "YEAR", "MONTH", "PRCP", "TAVG", "TMIN", "TMAX"]
    # mimic the processed historic daily read in for the historic monthly
    monthlyDict = {}
    primaryIdx = rawMonthlyDF["ID"] == primaryStationDict[station]
    primaryLat = float(list(set(rawMonthlyDF.loc[primaryIdx, "LAT"].values))[0])
    primaryLon = float(list(set(rawMonthlyDF.loc[primaryIdx, "LON"].values))[0])
    primaryElev = float(list(set(rawMonthlyDF.loc[primaryIdx, "ELEV"].values))[0])
    for i in range(len(rawMonthlyDF)):
        # copy the row info
        row = rawMonthlyDF.iloc[i].copy()
        
        # get the keying parameters from the row
        stationID, year, month = row["ID"], row["YEAR"], row["MONTH"]
        lat, lon, elev, prcp, tavg, tmin, tmax = row["LAT"], row["LON"], row["ELEV"], row["PRCP"], row["TAVG"], row["TMIN"], row["TMAX"]
        distBetweenStations = CalculateGeographicDistance(point1=[primaryLat, primaryLon, primaryElev], point2=[lat, lon, elev])

        # the key itself
        monthlyKey = (year, month)
        
        # if this not a primary station, bias correct  
        biasIndex = (biasDF["Primary"] == primaryStationDict[station]) & (biasDF["Secondary"] == stationID) & (biasDF["Month"] == month)
        if stationID != primaryStationDict[station]:
            # prcp bias correction
            prcpScale = biasDF.loc[biasIndex, "PRCP Scaling"].values
            prcpPOffset = biasDF.loc[biasIndex, "PRCP Primary Offset"].values
            prcpSOffset = biasDF.loc[biasIndex, "PRCP Secondary Offset"].values
            prcp = prcpScale * (prcp - prcpSOffset) + prcpPOffset 
            prcp = 0 if prcp < 2.54E-4 else prcp
            # temp bias correction
            tempScale = biasDF.loc[biasIndex, "TAVG Scaling"].values
            tempPOffset = biasDF.loc[biasIndex, "TAVG Primary Offset"].values
            tempSOffset = biasDF.loc[biasIndex, "TAVG Secondary Offset"].values
            tavg = tempScale * (tavg - tempSOffset) + tempPOffset 
            tmin = tempScale * (tmin - tempSOffset) + tempPOffset 
            tmax = tempScale * (tmax - tempSOffset) + tempPOffset 
       
        # special prcp case for Green Mt Dam after 2003 ... clever "hack" to deprioritize this station by setting distance to inf
        if (station == "Green Mt Dam") and (stationID == primaryStationDict[station]) and (year >= 2003):
            distBetweenStations = np.Inf
        # special prcp case for Eagle County after 2014 ... clever "hack" to prioritize this station by setting distance to < 0
        if (station == "Eagle County") and (stationID == "US1COEG0006") and (year >= 2014):
            distBetweenStations = -1.
        # special temp case for Yampa between 1976 and 1979 ... clever "hack" to prioritize this station by setting distance to < 0
        if (station == "Yampa") and (stationID == "USC00057936") and (year >= 1976 and year <= 1979):
            distBetweenStations = -1.

        # rules for how to add it to the dictionary!
        if monthlyKey not in monthlyDict.keys():
            # -- if we haven't seen this date before, simply add it
            monthlyDict[monthlyKey] = [station, primaryLat, primaryLon, primaryElev, year, month, prcp, tavg, tmin, tmax, distBetweenStations]
        else:         
            # -- check if the existing prcp is NaN, if so, fill with this value
            if np.isnan(monthlyDict[monthlyKey][6]) and not np.isnan(prcp):
                monthlyDict[monthlyKey][6] = prcp
            # -- check if the existing tavg is NaN, if so, fill with this value
            if np.isnan(monthlyDict[monthlyKey][7]) and not np.isnan(tavg):
                monthlyDict[monthlyKey][7] = tavg
                monthlyDict[monthlyKey][8] = tmin
                monthlyDict[monthlyKey][9] = tmax
            # -- check if this station is closer to the primary station
            if distBetweenStations < monthlyDict[monthlyKey][-1]:
                if not np.isnan(prcp):
                    monthlyDict[monthlyKey][6] = prcp
                    monthlyDict[monthlyKey][-1] = distBetweenStations
                if not np.isnan(tavg):
                    monthlyDict[monthlyKey][7] = tavg
                    monthlyDict[monthlyKey][8] = tmin
                    monthlyDict[monthlyKey][9] = tmax
                    monthlyDict[monthlyKey][-1] = distBetweenStations

    # convert from a dictionary to a dataframe
    monthlyDF = pd.DataFrame.from_dict(monthlyDict, orient="index", columns=[*dfColumns, "DIST"])
    monthlyDF = monthlyDF[[*dfColumns]]
    monthlyDF["PRCP"], monthlyDF["TAVG"], monthlyDF["TMIN"], monthlyDF["TMAX"] = \
        monthlyDF["PRCP"].astype(float), monthlyDF["TAVG"].astype(float), monthlyDF["TMIN"].astype(float), monthlyDF["TMAX"].astype(float) 
    monthlyDF = monthlyDF.reset_index(drop=True)
    # sort by year, month
    sortMonthlyDF = monthlyDF.copy()
    sortMonthlyDF["MONTH"] = sortMonthlyDF["MONTH"].apply(lambda x: int(monthNumDict[x]))
    sortMonthlyDF = sortMonthlyDF.sort_values(["YEAR", "MONTH"], ascending=[True, True])
    sortMonthlyDF["MONTH"] = sortMonthlyDF["MONTH"].apply(lambda x: iMonthNumDict["{:02d}".format(x)])
    monthlyDF = sortMonthlyDF
    
    # fill missing month(s) for precipitation and temperature with average monthly values across all years     
    for i in range(len(monthlyDF)):
        # get the row
        row = monthlyDF.iloc[i].copy()
        year, month, prcp, tavg, tmin, tmax = row["YEAR"], row["MONTH"], row["PRCP"], row["TAVG"], row["TMIN"], row["TMAX"]
        yearIdx, monthIdx = monthlyDF["YEAR"] == year, monthlyDF["MONTH"] == month
         
        # separate precipitation and temperatures
        for k, v in {"PRCP": prcp, "TAVG": tavg, "TMIN": tmin, "TMAX": tmax}.items():
            if np.isnan(v):
                monthlyDF.loc[yearIdx & monthIdx, k] = np.nanmean(monthlyDF.loc[monthIdx, k].values)
     
    # save the combined dataframe
    monthlyDF.to_csv(processedDir + r"/NOAA_UCRBMonthly_{}.csv".format(stationFromFile), index=False)


# bias correction for the daily data (major bottleneck to do that many pandas index calls each disaggregation iteration)
def CreateHistoricDailyMinT(dailyDF, biasesDF):
    # run down the raw daily data as a single list, bias correcting if secondary station
    dailyDict = {}
    primaryIdx = dailyDF["STATION"] == primaryStationDict[station]
    primaryLat = float(list(set(dailyDF.loc[primaryIdx, "LATITUDE"].values))[0])
    primaryLon = float(list(set(dailyDF.loc[primaryIdx, "LONGITUDE"].values))[0])
    primaryElev = float(list(set(dailyDF.loc[primaryIdx, "ELEVATION"].values))[0])
    for i in range(len(dailyDF)):
        row = dailyDF.iloc[i].copy()
        # reading in the raw data, so need to some transformations on the year/month/day/TMIN as above 
        stationID, date, minT, maxT = row["STATION"], row["DATE"], float(row["TMIN"]), float(row["TMAX"])
        # move to the next entry if this minT is NaN
        if np.isnan(minT) or np.isnan(maxT):
            continue 
        # dict key info
        splitDate = str(date).split("-")
        year, month, day = int(splitDate[0]), iMonthNumDict[splitDate[1]], int(splitDate[2])
        dailyKey = (station, year, month, day)
        # if this not a primary station, bias correct
        if stationID != primaryStationDict[station]:
            biasIndex = (biasesDF["Primary"] == primaryStationDict[station]) & (biasesDF["Secondary"] == stationID) & (biasesDF["Month"] == month)
            tempScale = biasesDF.loc[biasIndex, "TAVG Scaling"].values
            tempPOffset = biasesDF.loc[biasIndex, "TAVG Primary Offset"].values
            tempSOffset = biasesDF.loc[biasIndex, "TAVG Secondary Offset"].values
            minT = tempScale * (minT - tempSOffset) + tempPOffset
            maxT = tempScale * (maxT - tempSOffset) + tempPOffset
        # station distance info
        stationLat, stationLon, stationElev = float(row["LATITUDE"]), float(row["LONGITUDE"]), float(row["ELEVATION"])
        distBetweenStations = CalculateGeographicDistance(point1=[primaryLat, primaryLon, primaryElev], point2=[stationLat, stationLon, stationElev])
        # if we haven't seen this date before, add it
        if dailyKey not in dailyDict.keys():
            dailyDict[dailyKey] = [station, year, month, day, minT, maxT, distBetweenStations]
        else:
            # if we have seen this date, check that it's the closer station, and if so replace
            if distBetweenStations < dailyDict[dailyKey][-1]:
                dailyDict[dailyKey] = [station, year, month, day, minT, maxT, distBetweenStations]
    
    # reformatting the dailyDict as something hashable to speed the disaggregation up
    df = pd.DataFrame.from_dict(data=dailyDict, orient="index", columns=["NAME", "YEAR", "MONTH", "DAY", "TMIN", "TMAX", "DIST"])
    df = df[["NAME", "YEAR", "MONTH", "DAY", "TMIN", "TMAX"]]
    years = sorted(set(df["YEAR"]))
    biasDict = {(station, year): np.array([]) for year in years}
    for year in years:
        yearIdx, biasDictKey = df["YEAR"] == year, (station, int(year))
        # reduce to just the month/day/tmin values
        monthDayT = df.loc[yearIdx, ["MONTH", "DAY", "TMIN", "TMAX"]].copy()
        monthDayT[["TMIN", "TMAX"]] = monthDayT[["TMIN", "TMAX"]].astype(float)
        # sort bias corrected daily data from Jan1 to Dec31
        dtStr = monthDayT.apply(lambda x: str(int(year)) + "-" + monthNumDict[x["MONTH"]] + "-" + "{:02d}".format(x["DAY"]), axis=1)
        doys = np.array([pd.Period(dtstr, freq="D").day_of_year for dtstr in dtStr.values]) - 1
        monthDayT.index = doys
        monthDayT = monthDayT.sort_index()
        # add the sorted values to the biasDict at the appropriate key
        biasDict[biasDictKey] = monthDayT.values

    # save the new dictionary, return it
    # noinspection PyTypeChecker
    np.save(processedDir + r"/NOAA_UCRBDailyT_{}.npy".format(stationFromFile), biasDict)


# function for calculating distances on a spheroid
def CalculateGeographicDistance(point1, point2, convert2Radians=True):
    # define the radius of Earth (km)
    r = 6371.

    # if values need to be converted to radians, do so
    if convert2Radians:
        lon1 = np.deg2rad(point1[0])
        lat1 = np.deg2rad(point1[1])
        lon2 = np.deg2rad(point2[0])
        lat2 = np.deg2rad(point2[1])
    else:
        lon1 = point1[0]
        lat1 = point1[1]
        lon2 = point2[0]
        lat2 = point2[1]
    elev1 = point1[2] / 1e3
    elev2 = point2[2] / 1e3
    lonDiff = abs(lon2 - lon1)

    # arc length calculation
    num = np.sqrt(np.power(np.cos(lat2)*np.sin(lonDiff), 2.) + np.power(np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lonDiff), 2.))
    dem = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lonDiff)
    arcRad = np.arctan2(num, dem)
    arcLength = float(r * arcRad)

    # distance calculation
    return float(np.sqrt((arcLength ** 2.) + ((elev2 - elev1) ** 2.)))


# run it all when the script is called...
if __name__ == "__main__":
    monthlyRawDF = CreateHistoricRawMonthly()
    monthlyBiasDF = CreateHistoricMonthlyBias(monthlyRawDF)
    CreateHistoricProcessedUCRBDatasets(monthlyRawDF, monthlyBiasDF)


