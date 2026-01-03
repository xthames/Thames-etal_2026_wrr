# import
import os
import sys
import numpy as np
import pandas as pd
import StateCUDataReader


# environment variables
s, r = int(sys.argv[1]) + 1, int(sys.argv[2]) + 1


# filepaths
configsDir = os.path.dirname(os.path.dirname(__file__)) + r"/configs"
controlDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU"
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
analysisDir = os.path.dirname(os.path.dirname(__file__)) + r"/analysis"


# construct the raw crop key to crop title dict
def BuildCropDict():
    controlCDS = pd.read_csv(analysisDir + r"/CropInfo.csv")
    cropDict = {}
    for rawcrop in sorted(set(controlCDS["CROP"].values)):
        crop = rawcrop.split(".")[0]
        if "WITH_COVER" in crop:
            crop = "COVERED ORCHARD"
        if "WO_COVER" in crop:
            crop = "UNCOVERED_ORCHARD"
        cropVal = crop.replace("_", " ").title()
        cropDict[rawcrop] = cropVal
    return cropDict


# build the wx and gs dataframes
def BuildWXandGSIWR():
    # read in the elevation data
    elevDF = pd.read_csv(analysisDir + r"/ElevationInfo.csv")
    elevDF.astype({"WDID": str, "ELEV": float})
    wdids = sorted(set(elevDF["WDID"].values)) 
    
    # read in the wx data 
    controlWXDF = StateCUDataReader.ReadWXs("control")
    controlWXDtypes = {col: float for col in controlWXDF}
    controlWXDtypes["WDID"], controlWXDtypes["YEAR"] = str, int
    controlWXDF.astype(controlWXDtypes)
    
    # read in the gs data
    controlGSDF = StateCUDataReader.ReadOBC(controlDir + r"/cm2015B.obc")
    controlGSDF.astype({"WDID": str, "YEAR": int, "CROP": str, "GSSTART": int, "GSEND": int, "IWR": float})
    controlCDS = pd.read_csv(analysisDir + r"/CropInfo.csv")
    controlCDS.astype({"WDID": str, "YEAR": int, "AREA": float, "NCROPS": int, "CROP": str, "FRAC": float, "ACRES": float})

    # build per sow, realization
    wxDict, expGSDict = {}, {}
    # wx 
    realizationWX = pd.read_csv(syntheticDir + r"/CMIP6/Scenario{}/Sim{}/WX.csv".format(s, r))
    realizationWX.insert(loc=0, column="SOW", value=[s]*realizationWX.shape[0])
    realizationWX.insert(loc=1, column="REALIZATION", value=[r]*realizationWX.shape[0]) 
    realizationWXDtypes = {col: float for col in realizationWX.columns}
    realizationWXDtypes["SOW"], realizationWXDtypes["REALIZATION"], realizationWXDtypes["WDID"], realizationWXDtypes["YEAR"] = int, int, str, int
    realizationWX.astype(realizationWXDtypes)
    # gs
    realizationGS = pd.read_csv(syntheticDir + r"/CMIP6/Scenario{}/Sim{}/GS.csv".format(s, r))
    realizationGS.insert(loc=0, column="SOW", value=[s]*realizationGS.shape[0])
    realizationGS.insert(loc=1, column="REALIZATION", value=[r]*realizationGS.shape[0])
    realizationGS.astype({"SOW": int, "REALIZATION": int, "WDID": str, "YEAR": int, "GSSTART": int, "GSEND": int, "IWR": float})
    # for each wdid...
    for wdid in wdids:
        distr = int(wdid[:2])
        elev = round(elevDF.loc[elevDF["WDID"] == wdid, "ELEV"].values[0]/1000., 4)
        # setup
        ctrlWDIDIdx, expWDIDIdx = controlWXDF["WDID"] == wdid, realizationWX["WDID"] == wdid
        ctrlEntry, expEntry = controlWXDF.loc[ctrlWDIDIdx], realizationWX.loc[expWDIDIdx]
        prcpCtrlMean, tempCtrlMean = np.nanmean(ctrlEntry["PRCTOT"].values), np.nanmean(ctrlEntry["TEMAVG"].values)
        wdidExpGSIdx, wdidCDSIdx = realizationGS["WDID"] == wdid, controlCDS["WDID"] == wdid
        expGSEntry, cdsEntry = realizationGS.loc[wdidExpGSIdx], controlCDS.loc[wdidCDSIdx]
        years = sorted(set(expGSEntry["YEAR"].values))
        # wx
        for year in years:
            yearIdx = expEntry["YEAR"] == year
            yearEntry = expEntry.loc[yearIdx]
            prcpVal = round(yearEntry["PRCTOT"].values[0], 3)
            tempVal = round(yearEntry["TEMAVG"].values[0], 3)
            wxDict[(s, r, wdid, year)] = [wdid, distr, elev, s, r, year, prcpVal, tempVal]
        # gs
        for year in years:
            expYearIdx, cdsYearIdx = expGSEntry["YEAR"] == year, cdsEntry["YEAR"] == year
            expYearEntry, cdsYearEntry = expGSEntry.loc[expYearIdx], cdsEntry.loc[cdsYearIdx]
            crops = sorted(set(expYearEntry["CROP"].values))
            expYearCropDict = {}
            for crop in crops:
                cropKey = cropDict[crop]
                if cropKey not in expYearCropDict:
                    expYearCropDict[cropKey] = {"GS": [], "IWR": 0.}
                expCropIdx, cdsCropIdx = expYearEntry["CROP"] == crop, cdsYearEntry["CROP"] == crop
                expCropEntry, cdsCropEntry = expYearEntry.loc[expCropIdx], cdsYearEntry.loc[cdsCropIdx]
                expYearCropDict[cropKey]["GS"].append(expCropEntry["GSEND"].values[0] - expCropEntry["GSSTART"].values[0] + 1)
                cdsMult = cdsCropEntry["FRAC"].values[0] * cdsCropEntry["AREA"].values[0] 
                expYearCropDict[cropKey]["IWR"] += (expCropEntry["IWR"].values[0] / 12.) * cdsMult * AF_to_m3
            for crop in expYearCropDict.keys():
                expMeanGS = np.nanmean(expYearCropDict[crop]["GS"])
                expGSDict[(s, r, wdid, year, crop)] = [s, r, wdid, distr, elev, year, crop, expMeanGS, round(expYearCropDict[crop]["IWR"], 3)]
 
    # convert to DFs
    wxColumns=["WDID", "DISTRICT", "ELEV", "SOW", "REALIZATION", "YEAR", "PRCP", "TEMP"]
    wxDF = pd.DataFrame().from_dict(wxDict, orient="index", columns=wxColumns)
    wxDF.reset_index(drop=True, inplace=True)
    wxDtypes = {col: float for col in wxColumns}
    wxDtypes["WDID"] = str 
    wxDtypes["DISTRICT"], wxDtypes["SOW"], wxDtypes["REALIZATION"], wxDtypes["YEAR"] = int, int, int, int
    wxDF.astype(wxDtypes)
    wxDF.to_csv(analysisDir + r"/WXChange-S{}R{}.csv".format(s, r), index=False) 
    expTypes = {"SOW": int, "REALIZATION": int, "WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    gsExpDF = pd.DataFrame().from_dict(expGSDict, orient="index", columns=list(expTypes.keys()))
    gsExpDF.reset_index(drop=True, inplace=True)
    gsExpDF.astype(expTypes)
    gsExpDF.to_csv(analysisDir + r"/GSIWR-S{}R{}.csv".format(s, r), index=False) 
    

if __name__ == "__main__":
    AF_to_m3 = 1233.4818375
    cropDict = BuildCropDict()
    BuildWXandGSIWR()

