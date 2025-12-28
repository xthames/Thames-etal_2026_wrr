import os
import numpy as np
import pandas as pd
import datetime as dt


# filepaths
controlDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU"
noaaDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed/NOAA"
origPrcFP = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.prc"
origTemFP = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.tem"
origFdFP = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.fd"


# write the NOAA data to .prc, .tem, .fd files
def WritePrcTemFd():
    stationDict = {"Altenbern": "USC00050214", "Collbran": "USC00051741",
                   "Eagle County": "USW00023063", "Fruita": "USC00053146",
                   "Glenwood Springs": "USC00053359", "Grand Junction": "USC00053489",
                   "Grand Lake": "USC00053500", "Green Mt Dam": "USC00053592",
                   "Kremmling": "USC00054664", "Meredith": "USC00055507",
                   "Rifle": "USC00057031", "Yampa": "USC00059265"}
    monthNumDict = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}

    def WritePrcTem():
        # load in the data formatting lines from the StateCU .prc file
        prcDataFormatLine, temDataFormatLine = "", ""
        with (open(origPrcFP, "r") as prcFile, open(origTemFP, "r") as temFile):
            prcLines = prcFile.readlines()
            temLines = temFile.readlines()
        # #>EndHeader line tells us how StateCU will read in the data
        for l, line in enumerate(prcLines):
            if "#>EndHeader" in line:
                prcBreakLine = prcLines[l+1]
                prcColumnHeaderLine = prcLines[l+2]
                prcDataFormatLine = prcLines[l+3]
                prcUnitsLine = prcLines[l+4]
                break
        prcUnitsLine = prcUnitsLine.replace("1950", str(min(sorted(set(monthlyData["YEAR"])))))
        prcUnitsLine = prcUnitsLine.replace("2013", str(max(sorted(set(monthlyData["YEAR"])))))
        for l, line in enumerate(temLines):
            if "#>EndHeader" in line:
                temBreakLine = temLines[l+1]
                temColumnHeaderLine = temLines[l+2]
                temDataFormatLine = temLines[l+3]
                temUnitsLine = temLines[l+4]
                break
        temUnitsLine = temUnitsLine.replace("1950", str(min(sorted(set(monthlyData["YEAR"])))))
        temUnitsLine = temUnitsLine.replace("2013", str(max(sorted(set(monthlyData["YEAR"])))))
        # make a list of the b/e indices
        prcBs, prcEs, temBs, temEs = [0], [prcDataFormatLine.find("e")], [0], [temDataFormatLine.find("e")]
        prcB, prcE, temB, temE = prcBs[0], prcEs[0], temBs[0], temEs[0]
        while prcB >= 0 and prcE >= 0:
            prcB, prcE = prcDataFormatLine.find("b", prcE + 1), prcDataFormatLine.find("e", prcE + 1)
            prcBs.append(prcB), prcEs.append(prcE)
        while temB >= 0 and temE >= 0:
            temB, temE = temDataFormatLine.find("b", temE + 1), temDataFormatLine.find("e", temE + 1)
            temBs.append(temB), temEs.append(temE)
        prcBs, prcEs, temBs, temEs = prcBs[:-1], prcEs[:-1], temBs[:-1], temEs[:-1]

        # helper function for spacing out the data appropriately when writing
        def DataFormatter(val, b, e, frontPad=True):
            numMaxChars = e - b
            numDataChars = len(val)
            formattedData = ""
            if frontPad:
                for _ in range(numMaxChars - numDataChars + 1):
                    formattedData += " "
                formattedData += val
            else:
                formattedData += val
                for _ in range(numMaxChars - numDataChars):
                    formattedData += " "
            return formattedData

        # actually writing the files
        with (open(outputPrcFP, "w") as prcpFile, open(outputTemFP, "w") as tempFile):
            # define which simulation to use
            synthData = monthlyData

            # into, sub-header stuff for the .prc and .tem files
            prcpFile.write(prcBreakLine)
            prcpFile.write(prcColumnHeaderLine)
            prcpFile.write(prcDataFormatLine)
            prcpFile.write(prcUnitsLine)
            tempFile.write(temBreakLine)
            tempFile.write(temColumnHeaderLine)
            tempFile.write(temDataFormatLine)
            tempFile.write(temUnitsLine)

            # start reading in the data in the appropriate order
            for year in sorted(set(synthData["YEAR"])):
                yearIdx = synthData["YEAR"] == year
                for station in sorted(set(synthData["STATION"])):
                    stationIdx = synthData["STATION"] == station
                    # fundamental line to write, with year and station ID
                    prcpLine, tempLine = "", ""
                    prcpLine += DataFormatter(str(year), prcBs[0], prcEs[0]) + " "
                    prcpLine += DataFormatter(stationDict[station], prcBs[1], prcEs[1], frontPad=False) + " "
                    tempLine += DataFormatter(str(year), temBs[0], temEs[0]) + " "
                    tempLine += DataFormatter(stationDict[station], temBs[1], temEs[1], frontPad=False) + " "

                    # add in the monthly data
                    # -- precipitation
                    prcpVals = synthData.loc[yearIdx & stationIdx, "PRCP"].astype(float).values
                    if len(prcpVals) == 0: prcpVals = np.array(missingMonthlyDict["PRCP"][station])
                    prcpVals = prcpVals / 0.0254
                    for p, prcpVal in enumerate(prcpVals):
                        prcpLine += DataFormatter(str(-999.00), prcBs[2 + p], prcEs[2 + p]) if np.isnan(prcpVal) else DataFormatter("{:.2f}".format(prcpVal), prcBs[2 + p], prcEs[2 + p])
                    prcpLine += DataFormatter(str(-999.00), prcBs[2 + p + 1], prcEs[2 + p + 1]) + "\n" if np.all(np.isnan(prcpVals)) else DataFormatter("{:.2f}".format(np.nansum(prcpVals)), prcBs[2 + p + 1], prcEs[2 + p + 1]) + "\n"
                    prcpFile.write(prcpLine)

                    # -- temperature
                    tempVals = synthData.loc[yearIdx & stationIdx, "TAVG"].astype(float).values
                    if len(tempVals) == 0: tempVals = np.array(missingMonthlyDict["TAVG"][station])
                    tempVals = tempVals * (9. / 5.) + 32.
                    for t, tempVal in enumerate(tempVals):
                        tempLine += DataFormatter(str(-999.00), temBs[2 + t], temEs[2 + t]) if np.isnan(tempVal) else DataFormatter("{:.2f}".format(tempVal), temBs[2 + t], temEs[2 + t])
                    tempLine += DataFormatter(str(-999.00), temBs[2 + t + 1], temEs[2 + t + 1]) + "\n" if np.all(np.isnan(tempVals)) else DataFormatter("{:.2f}".format(np.nanmean(tempVals)), temBs[2 + t + 1], temEs[2 + t + 1]) + "\n"
                    tempFile.write(tempLine)

    def WriteFd():
        def CreateFdDF():
            fdDict = {}
            for k in dailyTDict:
                station, year = k[0], k[1]
                sp, fa = [], []
                for i in range(len(dailyTDict[k])):
                    month = dailyTDict[k][i][0]
                    day = dailyTDict[k][i][1]
                    doy = int(dt.datetime.strptime("{}-{}-{}".format(year, month, day), "%Y-%b-%d").strftime("%j"))
                    tmin = (dailyTDict[k][i][2] * (9./5)) + 32.  
                    if doy < 182:
                        sp.append([doy, tmin])
                    else:
                        fa.append([doy, tmin])
                sp = sp[::-1]
                fds = []
                for season in [sp, fa]:
                    for threshold in [28, 32]:
                        i = 0
                        while i < len(season):
                            if season[i][1] < threshold:
                                fds.append(dt.datetime.strptime(str(year) + "-" + str(season[i][0]), "%Y-%j").strftime("%m/%d"))
                                break
                            i += 1
                        if i == len(season):
                            fds.append("-999.0") 
                fdDict[(year, station)] = [year, station, fds[0], fds[1], fds[3], fds[2]]
            return pd.DataFrame.from_dict(data=fdDict, orient="index", columns=["YEAR", "STATION", "LAST SPR 28", "LAST SPR 32", "FIRST FALL 32", "FIRST FALL 28"])
 
        # formatting fd data according to the StateCU historic input
        fdDataFormatLine = ""
        with open(origFdFP, "r") as hfdFile:
            fdLines = hfdFile.readlines()
        # #>EndHeader line tells us how StateCU will read in the data
        for l, line in enumerate(fdLines):
            if "#>Temperatures are degrees F" in line:
                fdBreakLine = fdLines[l+1]
                fdColumnHeaderLine1 = fdLines[l+2]
                fdColumnHeaderLine2 = fdLines[l+3]
                fdDataFormatLine = fdLines[l+4]
                fdUnitsLine = fdLines[l+6]
                break
        fdUnitsLine = fdUnitsLine.replace("1950", str(min(sorted(set(monthlyData["YEAR"])))))
        fdUnitsLine = fdUnitsLine.replace("2013", str(max(sorted(set(monthlyData["YEAR"])))))
        # make a list of the b/e indices
        fdBs, fdEs = [0], [fdDataFormatLine.find("e")]
        fdB, fdE = fdBs[0], fdEs[0]
        while fdB >= 0 and fdE >= 0:
            fdB, fdE = fdDataFormatLine.find("b", fdE + 1), fdDataFormatLine.find("e", fdE + 1)
            fdBs.append(fdB), fdEs.append(fdE)
        fdBs, fdEs = fdBs[:-1], fdEs[:-1]

        # helper function for spacing out the data appropriately when writing
        def DataFormatter(val, b, e, frontPad=True):
            numMaxChars = e - b
            numDataChars = len(val)
            formattedData = ""
            if frontPad:
                for _ in range(numMaxChars - numDataChars + 1):
                    formattedData += " "
                formattedData += val
            else:
                formattedData += val
                for _ in range(numMaxChars - numDataChars):
                    formattedData += " "
            return formattedData

        synthData = monthlyData
        frostDatesDF = CreateFdDF()
        # fill missing with frostdate averages
        missingFDDict = {}
        for station in sorted(set(frostDatesDF["STATION"].values)):
            missingFDDict[station] = {}
            stationIdx = frostDatesDF["STATION"] == station
            stationEntry = frostDatesDF.loc[stationIdx]
            statecuStationEntry = stationEntry.loc[(stationEntry["YEAR"] >= 1950) & (stationEntry["YEAR"] <= 2013)]
            for col in ["LAST SPR 28", "LAST SPR 32", "FIRST FALL 32", "FIRST FALL 28"]:
                doys = []
                for i in range(statecuStationEntry.shape[0]):
                    rowEntry = statecuStationEntry.iloc[i]
                    year, colDate = rowEntry["YEAR"], rowEntry[col]
                    if colDate == "-999.0": continue
                    doy = int(dt.datetime.strptime("{}/{}".format(year, colDate), "%Y/%m/%d").strftime("%j"))
                    doys.append(doy)
                missingFDDict[station][col] = round(np.nanmean(doys))

        with open(outputFdFP, "w") as fdFile:
            fdFile.write(fdBreakLine)
            fdFile.write(fdColumnHeaderLine1)
            fdFile.write(fdColumnHeaderLine2)
            fdFile.write(fdDataFormatLine)
            fdFile.write(fdUnitsLine)

            # start reading in the data in the appropriate order
            for yr in sorted(set(synthData["YEAR"])):
                yrIdx = frostDatesDF["YEAR"] == yr
                for stn in sorted(set(synthData["STATION"])):
                    stnIdx = frostDatesDF["STATION"] == stn
                    fdLine = ""
                    fdLine += DataFormatter(str(yr), fdBs[0], fdEs[0]) + " "
                    fdLine += DataFormatter(stationDict[stn], fdBs[1], fdEs[1], frontPad=False) + " "
                    for cn, col in enumerate(["LAST SPR 28", "LAST SPR 32", "FIRST FALL 32", "FIRST FALL 28"]):
                        colEntry = frostDatesDF.loc[yrIdx & stnIdx, col]
                        if colEntry.empty or colEntry.values[0] == "-999.0":
                            fdVal = dt.datetime.strptime("{}-{}".format(yr, missingFDDict[stn][col]), "%Y-%j").strftime("%m/%d")
                        else:
                            fdVal = frostDatesDF.loc[yrIdx & stnIdx, col].values[0]
                        fdLine += DataFormatter(fdVal, fdBs[2 + cn], fdEs[2 + cn])
                    fdLine += "\n"
                    fdFile.write(fdLine)
    
    # functions
    WritePrcTem()
    WriteFd()


# execute
if __name__ == "__main__":
    # load data
    monthlyData = pd.read_csv(noaaDir + r"/NOAA_UCRBMonthly.csv")
    monthlyData.rename(columns={"NAME": "STATION"}, inplace=True)
    dailyTDict = np.load(noaaDir + r"/NOAA_UCRBDailyT.npy", allow_pickle=True).item()

    # dictionary for missing months with averages
    missingMonthlyDict = {"PRCP": {}, "TAVG": {}}
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        monthIdx = monthlyData["MONTH"] == month
        monthEntry = monthlyData.loc[monthIdx]
        for station in sorted(set(monthEntry["STATION"].values)):
            stationIdx = monthEntry["STATION"] == station
            stationEntry = monthEntry.loc[stationIdx]
            statecuStationEntry = stationEntry.loc[(stationEntry["YEAR"] >= 1950) & (stationEntry["YEAR"] <= 2013)]
            avgPrcp, avgTemp = np.nanmean(statecuStationEntry["PRCP"].values), np.nanmean(statecuStationEntry["TAVG"].values)
            for wvar, avg in zip(["PRCP", "TAVG"], [avgPrcp, avgTemp]):
                if station not in missingMonthlyDict[wvar]:
                    missingMonthlyDict[wvar][station] = [avg]
                else:
                    missingMonthlyDict[wvar][station].append(avg)

    # name filepaths
    outputPrcFP = controlDir + r"/COclim_NOAA.prc"
    outputTemFP = controlDir + r"/COclim_NOAA.tem"
    outputFdFP = controlDir + r"/COclim_NOAA.fd"
     
    # write
    WritePrcTemFd()
