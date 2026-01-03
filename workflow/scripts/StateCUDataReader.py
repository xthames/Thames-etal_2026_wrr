# imports
import os
import sys
import numpy as np
import pandas as pd
import datetime as dt


# filepaths
controlDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU"
templateDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/TemplateCU"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
analysisDir = os.path.dirname(os.path.dirname(__file__)) + r"/analysis"


# discover some interesting users from the UCRB
# -- largest/smallest IWR
# -- largest/smallest area
# ##-- similar users, but high vs. low elevation
# ##-- similar users, but flood vs. sprinkler irrigation
def FindInterestingUsers():
    interestingCriteriaDict = {}
    
    # --> SMALLEST, MEDIAN, LARGEST USERS
    # read in from control .cir (for IWR, area)
    cirDict = ReadCIR(controlDir + r"/cm2015B.cir")
    
    # find the largest, median, smallest amounts as averages across all years
    userIWRAmounts = []
    for userID in cirDict:
        # access the DF, find the averages
        df = cirDict[userID]
        IWRs = df["ANNUAL"].values
        userIWRAmounts.append((np.nanmean(IWRs[IWRs != 0.]), userID))
    userIWRAmounts.sort(key=lambda x: x[0])

    # fill
    interestingCriteriaDict["iwr"] = {}
    interestingCriteriaDict["iwr"]["smallestIWR"] = [*userIWRAmounts[0]]
    interestingCriteriaDict["iwr"]["medianIWR"] = [*userIWRAmounts[len(userIWRAmounts)//2]]
    interestingCriteriaDict["iwr"]["largestIWR"] = [*userIWRAmounts[-1]]
    

    # --> ELEVATION (manual checking: largest users at lowest, highest elevations)
    interestingCriteriaDict["elevation"] = {"low": "7200646_I", "high": "5100848"}     

    # return
    return interestingCriteriaDict 


# read in the .cir file
def ReadCIR(fp):
    # base read
    with open(fp, "r") as f:
        data = f.readlines()
    
    # for lines in the data...
    iwrDict, l, months = {}, 0, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cirColumns = ["YEAR", "AREA", *[month.upper() for month in months], "ANNUAL", "AF/ac"]
    while l < len(data):
        line = data[l]
        # -- start an extraction when you see a line start with "_"
        if line[0] == "_":
            sep = line.index(" ")
            # -- identify the parcel/user
            parcelID = line[1:sep]

            # -- it's 6 lines from the parcel/user ID to the start of the data
            ll, dataDict = 0, {}
            while data[l+6+ll][0] != "-":
                i = l+6+ll
                line = data[i].strip().replace("\n", "")
                # -- the first 4 characters are the year     
                year, extras = int(line[:4]), line[4:]
                # -- extracting the area, IWR by month, total, AF/ac
                splitExtras = [extra.strip() for extra in extras.split(".")]
                splitExtras[-2] = splitExtras[-2] + "." + splitExtras[-1] 
                splitExtras.pop(-1)
                dataDict[year] = [year, *[float(splitExtra) for splitExtra in splitExtras]]
                ll += 1
            
            # -- turn the dictionary into a DF, add it to the full IWR dict
            df = pd.DataFrame.from_dict(dataDict, orient="index", columns=cirColumns)
            df.reset_index(drop=True, inplace=True)
            iwrDict[parcelID] = df

            # -- jump to the next section 
            l += 6+ll
        else: 
            # -- otherwise just increment the line counter
            l += 1
    
    # return the dictionary with all parcel/user IWR information
    return iwrDict


# read in the .cds file
def ReadCDS():
    # base read
    with open(controlDir + r"/cm2015B.cds", "r") as f:
        data = f.readlines()

    # columns from the cds files
    cdsColumns = ["WDID", "YEAR", "AREA", "NCROPS", "CROP", "FRAC", "ACRES"]
    
    # reading each line
    cropDict, l = {}, 0
    while l < len(data):
        line = data[l]
        # -- don't read the header
        if line[0] == "#": pass
        # -- if the start of the line is numeric, it's the start of a year so read it in
        if line[0].isnumeric():
            lineData = line.replace("\n", "").split(" ")
            lineData = [e for e in lineData if e != ""]
            year, wdid, area, ncrops = int(lineData[0]), lineData[1], float(lineData[2]), int(lineData[3])
            for ll in range(1, ncrops+1):
                subLine = data[l+ll]
                subLineData = subLine.replace("\n", "").split(" ")
                subLineData = [e for e in subLineData if e != ""]
                crop, frac, acres = subLineData[0], float(subLineData[1]), float(subLineData[2])
                cropDict[(wdid, year, crop)] = [wdid, year, area, ncrops, crop, frac, acres]
        l += 1
    
    # convert the dictionary to a dataframe
    cdsDF = pd.DataFrame().from_dict(cropDict, orient="index", columns=cdsColumns)
    cdsDF.reset_index(drop=True, inplace=True)
    cdsDF.astype({"WDID": str, "YEAR": int, "AREA": float, "NCROPS": int, "CROP": str, "FRAC": float, "ACRES": float})

    # return
    return cdsDF


# read in the .ipy file
def ReadIPY():
    # base read
    with open(controlDir + r"/cm2015B.ipy", "r") as f:
        data = f.readlines()

    # for lines in the data...
    irriDict, l = {}, 0 
    ipyColumns = ["YEAR", "ID", "SURF EFF", "FLD EFF", "SPR EFF", "AC FLD ONLY", "AC SPR ONLY", "AC FLD+GRD", "AC SPR+GRD", "PUMP MAX", "GMODE", "TOTAL AC", "AC SURF ONLY", "AC GRD ONLY"]
    while l < len(data):
        if data[l][0] != "#": 
            # strip, replace the newline with nothing, remove the whitespace and separate
            line = data[l].strip().replace("\n", "")
            sepLine = [el for el in line.split() if el != ""]
            if (len(sepLine) == len(ipyColumns)) and ((int(sepLine[0]) >= 1950) and (int(sepLine[0]) <= 2013)):
                irriDict[l] = sepLine
        l += 1
    
    # convert the irriDict to an irriDF
    irriDtypes = {col: float for col in ipyColumns}
    irriDtypes["YEAR"], irriDtypes["ID"], irriDtypes["GMODE"] = int, str, int
    rawIrriDF = pd.DataFrame().from_dict(irriDict, orient="index", columns=ipyColumns)
    rawIrriDF = rawIrriDF.astype(irriDtypes)
    
    # determine what fraction of the total acreage is flood, sprinkler per year for each user
    irriFracDict = {}
    for userID in sorted(set(rawIrriDF["ID"].values)):
        userIdx = rawIrriDF["ID"] == userID
        rawIrriEntry = rawIrriDF.loc[userIdx]
        for year in rawIrriEntry["YEAR"].values:
            yearIdx = rawIrriEntry["YEAR"] == year
            totalAcres = rawIrriEntry.loc[yearIdx, "TOTAL AC"].values
            if totalAcres == 0.:
                floodedFrac, sprinkledFrac = 0., 0.
            else:
                floodedAcres = rawIrriEntry.loc[yearIdx, "AC FLD ONLY"].values + rawIrriEntry.loc[yearIdx, "AC FLD+GRD"].values 
                sprinkledAcres = rawIrriEntry.loc[yearIdx, "AC SPR ONLY"].values + rawIrriEntry.loc[yearIdx, "AC SPR+GRD"].values  
                floodedFrac, sprinkledFrac = floodedAcres / totalAcres, sprinkledAcres / totalAcres
            irriFracDict[(userID, year)] = [userID, year, totalAcres, floodedFrac, sprinkledFrac]
    irriFracDF = pd.DataFrame().from_dict(irriFracDict, orient="index", columns=["WDID", "YEAR", "AREA", "FLOOD FRAC", "SPRINKLER FRAC"])
    irriFracDF = irriFracDF.astype({"WDID": str, "YEAR": int, "AREA": float, "FLOOD FRAC": float, "SPRINKLER FRAC": float})
    irriFracDF.reset_index(drop=True, inplace=True)

    # return
    return irriFracDF


# read the .cli file
def ReadCLI():
    # base read
    with open(templateDir + r"/simulation.cli", "r") as f:
        data = f.readlines()
    
    # for lines in the data...
    cliDict, l = {}, 0 
    cliColumns = ["ID", "LAT", "ELEV", "REGION1", "REGION2", "NAME"]
    while l < len(data):
        if data[l][0] != "#": 
            # strip, replace the newline with nothing, remove the whitespace and separate
            line = data[l].strip().replace("\n", "")
            sepLine = [el for el in line.split() if el != ""]
            if len(sepLine) > 6:
                joinedName = " ".join(sepLine[5:])
                cliDict[sepLine[0]] = [*sepLine[:5], joinedName]
            else:
                cliDict[sepLine[0]] = sepLine
        l += 1
    cliDF = pd.DataFrame().from_dict(cliDict, orient="index", columns=cliColumns)
    cliDF = cliDF.astype({"ID": str, "LAT": float, "ELEV": float, "REGION1": str, "REGION2": int, "NAME": str})
    cliDF.reset_index(drop=True, inplace=True)
    cliDF["ELEV"] = cliDF["ELEV"].values * 0.3048

    # return
    return cliDF


# read the .str file to extract WDID vs elevation
def ReadSTR1():
    # base read
    with open(controlDir + r"/cm2015.str", "r") as f:
        data = f.readlines()
    
    # read in the .cli file
    cliDF = ReadCLI()

    # for lines in the data...
    strDict, l = {}, 0 
    while l < len(data):
        if data[l][0] != "#": 
            # strip, replace the newline with nothing, remove the whitespace and separate
            line = data[l].strip().replace("\n", "")
            sepLine = [el for el in line.split() if el != ""]
            if data[l][0].isnumeric():
                parcelID = sepLine[0]
                strDict[parcelID] = []
            if data[l][0] == "U":
                strDict[parcelID].append(sepLine[0])
        l += 1
    
    # dataframe for each parcel, (average) elevation
    parcelElevDict = {}
    for parcelID, cliStations in strDict.items():
        elevs = [cliDF.loc[cliDF["ID"] == cliStation, "ELEV"].values for cliStation in cliStations]
        parcelElevDict[parcelID] = [parcelID, np.mean(elevs)] 
    parcelElevDF = pd.DataFrame().from_dict(parcelElevDict, orient="index", columns=["WDID", "ELEV"])
    parcelElevDF = parcelElevDF.astype({"WDID": str, "ELEV": float})
    parcelElevDF.reset_index(drop=True, inplace=True)

    # return
    return parcelElevDF


# read the .str file to extract WDID vs climate station weighting info
def ReadSTR2():
    # base read: .str
    with open(controlDir + r"/cm2015.str", "r") as f:
        data = f.readlines() 
    # for lines in the data...
    strDict, l = {}, 0 
    while l < len(data):
        if data[l][0] != "#": 
            # strip, replace the newline with nothing, remove the whitespace and separate
            line = data[l].strip().replace("\n", "")
            sepLine = [el for el in line.split() if el != ""]
            if data[l][0].isnumeric():
                parcelID = sepLine[0]
                strDict[parcelID] = {}
            if data[l][0] == "U":
                strDict[parcelID][sepLine[0]] = [float(sepLine[2]), float(sepLine[1])]
        l += 1
    
    # return
    return strDict


# aggregate .fd, .str, .tem files into growing season information
def ReadWXs(repo):
    # read in the .cli file
    cliDF = ReadCLI()
     
    # base read: .str
    with open(controlDir + r"/cm2015.str", "r") as f:
        data = f.readlines() 
    # for lines in the data...
    strDict, l = {}, 0 
    while l < len(data):
        if data[l][0] != "#": 
            # strip, replace the newline with nothing, remove the whitespace and separate
            line = data[l].strip().replace("\n", "")
            sepLine = [el for el in line.split() if el != ""]
            if data[l][0].isnumeric():
                parcelID = sepLine[0]
                strDict[parcelID] = {}
            if data[l][0] == "U":
                strDict[parcelID][sepLine[0]] = [float(sepLine[2]), float(sepLine[1])]
        l += 1
    
    wxCols = ["YEAR", "STATION", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANNUAL"]
    def ReadWX(wvar):
        wvarExt = "prc" if wvar == "PRCP" else "tem"
        wxFile = syntheticDir + r"/CMIP6/Scenario{}/Sim{}/simulation.{}".format(nSOW, nRealization, wvarExt) if repo == "exp" else controlDir + r"/COclim_NOAA.{}".format(wvarExt)
        # base read:
        with open(wxFile, "r") as f:
            data = f.readlines() 
        # dict info 
        wxDict, l = {}, 0
        while l < len(data):
            line = data[l]
            if (line[0] != "#") and (line[0].isnumeric()):
                lineData = [e for e in line.replace("\n", "").split(" ") if e != ""]
                year, station = int(lineData[0]), lineData[1]
                wxData = []
                for val in lineData[2:]:
                    if val == "-999.0":
                        wxDatum = np.nan
                    else:
                        wxDatum = float(val) * 0.0254 if wvar == "PRCP" else (float(val)-32.)*(5./9.)
                    wxData.append(wxDatum)
                wxDict[(year, station)] = [year, station, *wxData]
            l += 1
        # df info
        wxDF = pd.DataFrame().from_dict(wxDict, orient="index", columns=wxCols)
        wxDF.reset_index(drop=True, inplace=True)
        # return
        return wxDF

    # read the .prc, .tem files
    prcDF, temDF = ReadWX("PRCP"), ReadWX("TEMP")

    # dataframe of weather vars by WDID, weighted to climate station structure, as annualized precip and temp
    wdids, years, syncDict = list(strDict.keys()), sorted(set(prcDF["YEAR"])), {}
    for wdid in wdids:
        wdidStations = list(strDict[wdid].keys())
        prcpWeights = [strDict[wdid][stationID][0] for stationID in wdidStations]
        tempWeights = [strDict[wdid][stationID][1] for stationID in wdidStations]
        for year in years:
            prcYear, temYear = prcDF["YEAR"] == year, temDF["YEAR"] == year
            prcptot, tempavg = [], []
            for station in wdidStations:
                prcStation, temStation = prcDF["STATION"] == station, temDF["STATION"] == station
                prcStationEntry, temStationEntry = prcDF.loc[prcYear & prcStation], temDF.loc[temYear & temStation]
                prcptot.append(float(prcStationEntry["ANNUAL"].values[0]))
                tempavg.append(float(temStationEntry["ANNUAL"].values[0]))
            wdidPRCP = np.nan if any(np.isnan(prcptot)) else 100.*np.average(prcptot, weights=prcpWeights)
            wdidTEMP = np.nan if any(np.isnan(tempavg)) else np.average(tempavg, weights=tempWeights)
            syncDict[wdid, year] = [wdid, year, wdidPRCP, wdidTEMP]
    syncDF = pd.DataFrame().from_dict(syncDict, orient="index", columns=["WDID", "YEAR", "PRCTOT", "TEMAVG"])
    syncDF.reset_index(drop=True, inplace=True)
    syncDF.astype({"WDID": str, "YEAR": int, "PRCTOT": float, "TEMAVG": float})
    
    # return
    return syncDF


# read in the .obc file
def ReadOBC(fp):
    # base read
    with open(fp, "r") as f:
        data = f.readlines()
    
    # for lines in the data...
    obcDict, l = {}, 0
    wdid, year, crop, start, end, iwr = None, None, None, None, None, None
    monthChunks, monthFlag = [], False
    while l < len(data):
        # remove newlines, strip spaces
        line = data[l].replace("\n", "").strip() 
        monthChunkCheck = False

        # if newlines/spaces were everything on the line, we can skip it
        if len(line) == 0: 
            l += 1
            continue
        
        # chunking the data in the line by spaces
        lineChunks = [e for e in line.split(" ") if e != ""]

        # if the first character of the line is a number, it's the parcel ID
        if lineChunks[0][0].isnumeric():
            wdid = lineChunks[0]
        
        # if the last chunk has "Year=", extract year and crop
        if "Year" in lineChunks[-1]:
            crop = lineChunks[0]
            year = int(lineChunks[-1].split("=")[1])
            
        # find the line with all "-"
        if (len(lineChunks) == 1) and (all([True if s == "-" else False for s in lineChunks[0]])):
            monthFlag = not monthFlag
        if monthFlag:
            monthChunks.append(lineChunks)

        # if you've got Season in the first chunk, append month data and push what you have to the dictionary
        if lineChunks[0] == "Season":
            iwr = float(lineChunks[-1]) 
            monthChunks.pop(0)
            if len(monthChunks) <= 1:
                obcDict[(wdid, year, crop)] = [wdid, year, crop, 0, 0, iwr]
            else:
                # check to see if the start/end dates are empty
                emptyStart, emptyEnd, ed = False, False, "01"
                while float(monthChunks[0][-3]) == 0.:
                    monthChunks.pop(0)
                    emptyStart = True
                if float(monthChunks[-1][-3]) == 0.:
                    emptyEnd = True
                # GS start
                if emptyStart:
                    em = monthChunks[0][0]
                    start = int(dt.datetime.strptime("{}-{}-{}".format(year, em, ed), "%Y-%b-%d").strftime("%j"))
                else:
                    startChunk = monthChunks[0][:2]
                    if "." not in startChunk[1]:
                        m, d = startChunk[0], "0"+startChunk[1]
                    else:
                        m, d = startChunk[0][:3], startChunk[0][3:] 
                    start = int(dt.datetime.strptime("{}-{}-{}".format(year, m, d), "%Y-%b-%d").strftime("%j"))
                # GS end
                if emptyEnd:
                    em = monthChunks[-2][0]
                    end = int((dt.datetime.strptime("{}-{}-{}".format(year, em, ed), "%Y-%b-%d") - dt.timedelta(days=1)).strftime("%j")) 
                else:
                    endChunk = monthChunks[-1][:2]
                    if "." not in endChunk[1]:
                        m, d = endChunk[0], "0"+endChunk[1]
                    else:
                        m, d = endChunk[0][:3], endChunk[0][3:] 
                    end = int(dt.datetime.strptime("{}-{}-{}".format(year, m, d), "%Y-%b-%d").strftime("%j"))
                obcDict[(wdid, year, crop)] = [wdid, year, crop, start, end, iwr]
            monthChunks = []
        # increment
        l += 1
    gsDF = pd.DataFrame().from_dict(obcDict, orient="index", columns=["WDID", "YEAR", "CROP", "GSSTART", "GSEND", "IWR"]) 
    gsDF.reset_index(drop=True, inplace=True)
    gsDF.astype({"WDID": str, "YEAR": int, "CROP": str, "GSSTART": int, "GSEND": int, "IWR": float})
    return gsDF


# actual program
if __name__ == "__main__":
    # correctly ID SOW, realizations
    nSOW, nRealization = int(sys.argv[1]) + 1, int(sys.argv[2]) + 1

    # read in the data
    #iwrData = ReadCIR(syntheticDir + r"/CMIP6/Scenario{}/Sim{}/simulation.cir".format(nSOW, nRealization))
    wxData = ReadWXs("exp")
    gsData = ReadOBC(syntheticDir + r"/CMIP6/Scenario{}/Sim{}/simulation.obc".format(nSOW, nRealization))
    
    # save the data
    #np.save(syntheticDir + r"/CMIP6/Scenario{}/Sim{}/IWR.npy".format(nSOW, nRealization), iwrData) 
    wxData.to_csv(syntheticDir + r"/CMIP6/Scenario{}/Sim{}/WX.csv".format(nSOW, nRealization), index=False)
    gsData.to_csv(syntheticDir + r"/CMIP6/Scenario{}/Sim{}/GS.csv".format(nSOW, nRealization), index=False)

