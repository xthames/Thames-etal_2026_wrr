# imports
import os
import sys
import numpy as np
import StateCUDataReader
import pandas as pd


# filepaths
scriptsDir = os.path.dirname(__file__)
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
analysisDir = os.path.dirname(os.path.dirname(__file__)) + r"/analysis"
controlDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU"


# environment vars
statecuAbortReport = sys.argv[1] if len(sys.argv) > 1 else ""


# get the number of SOWs, realizations
def DetermineSOWandRealizationNumber():
    # SOW number
    cmip6SynthDirData = os.listdir(syntheticDir + r"/CMIP6")
    sowDirs = [d for d in cmip6SynthDirData if os.path.isdir(syntheticDir + r"/CMIP6/{}".format(d))]
    
    # realization number
    cmip6SOWDirData = os.listdir(syntheticDir + r"/CMIP6/{}".format(sowDirs[0]))
    realizationDirs = [d for d in cmip6SOWDirData if os.path.isdir(syntheticDir + r"/CMIP6/{}/{}".format(sowDirs[0], d))]
    
    # return
    return len(sowDirs), len(realizationDirs)


# check for if StateCU threw a processing error and aborted
def CheckForAbortedStateCU():
    nErrors = 0
    for nSOW in range(1, nSOWs+1):
        for nRealization in range(1, nRealizations+1):
            rlzDir = syntheticDir + r"/CMIP6/Scenario{}/Sim{}".format(nSOW, nRealization)
            if "simulation.cir" not in os.listdir(rlzDir):
                nErrors += 1 
                with open(rlzDir + r"/simulation.obc") as f:
                    errorParcel = f.readlines()[-2].replace("\n", "")
                print("ERROR IN: SOW{}, RZ{} | {}".format(nSOW, nRealization, errorParcel))
    if nErrors == 0:
        print("!! STATECU SUCCESSFULLY IMPLEMENTED !!")
    else:
        print("ERRORS FOUND: {} | {}% of total".format(nErrors, round(100.*(nErrors/(nSOWs*nRealizations)), 2)))


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


# build the control GSIWR dataframe
def BuildControlGSIWR():
    # read in the elevation data
    elevDF = pd.read_csv(analysisDir + r"/ElevationInfo.csv")
    elevDF.astype({"WDID": str, "ELEV": float})
    wdids = sorted(set(elevDF["WDID"].values)) 
    cropDict = BuildCropDict()
    AF_to_m3 = 1233.4818375

    # read in the gs data
    controlGSDF = StateCUDataReader.ReadOBC(controlDir + r"/cm2015B.obc")
    controlGSDF.astype({"WDID": str, "YEAR": int, "CROP": str, "GSSTART": int, "GSEND": int, "IWR": float})
    controlCDS = pd.read_csv(analysisDir + r"/CropInfo.csv")
    controlCDS.astype({"WDID": str, "YEAR": int, "AREA": float, "NCROPS": int, "CROP": str, "FRAC": float, "ACRES": float})

    # build just the control GS outside
    ctrlGSDict = {}
    for wdid in wdids:
        distr = int(wdid[:2])
        elev = elevDF.loc[elevDF["WDID"] == wdid, "ELEV"].values[0]/1000.
        wdidCtrlGSIdx, wdidCDSIdx = controlGSDF["WDID"] == wdid, controlCDS["WDID"] == wdid
        ctrlGSEntry, cdsEntry = controlGSDF.loc[wdidCtrlGSIdx], controlCDS.loc[wdidCDSIdx]
        years = sorted(set(ctrlGSEntry["YEAR"].values))
        for year in years:
            ctrlYearIdx, cdsYearIdx = ctrlGSEntry["YEAR"] == year, cdsEntry["YEAR"] == year
            ctrlYearEntry, cdsYearEntry = ctrlGSEntry.loc[ctrlYearIdx], cdsEntry.loc[cdsYearIdx]
            crops = sorted(set(ctrlYearEntry["CROP"].values))
            ctrlYearCropDict = {}
            for crop in crops:
                cropKey = cropDict[crop]
                if cropKey not in ctrlYearCropDict:
                    ctrlYearCropDict[cropKey] = {"GS": [], "IWR": 0.}
                ctrlCropIdx, cdsCropIdx = ctrlYearEntry["CROP"] == crop, cdsYearEntry["CROP"] == crop
                ctrlCropEntry, cdsCropEntry = ctrlYearEntry.loc[ctrlCropIdx], cdsYearEntry.loc[cdsCropIdx]
                ctrlYearCropDict[cropKey]["GS"].append(ctrlCropEntry["GSEND"].values[0] - ctrlCropEntry["GSSTART"].values[0] + 1)
                cdsMult = cdsCropEntry["FRAC"].values[0] * cdsCropEntry["AREA"].values[0] 
                ctrlYearCropDict[cropKey]["IWR"] += (ctrlCropEntry["IWR"].values[0] / 12.) * cdsMult * AF_to_m3
            for crop in ctrlYearCropDict.keys():
                ctrlMeanGS = np.nanmean(ctrlYearCropDict[crop]["GS"])
                ctrlGSDict[(wdid, year, crop)] = [wdid, distr, elev, year, crop, ctrlMeanGS, round(ctrlYearCropDict[crop]["IWR"], 3)]
    ctrlTypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    gsCtrlDF = pd.DataFrame().from_dict(ctrlGSDict, orient="index", columns=list(ctrlTypes.keys()))
    gsCtrlDF.reset_index(drop=True, inplace=True)
    gsCtrlDF.astype(ctrlTypes)
    return gsCtrlDF


# actual program
if __name__ == "__main__":
    nSOWs, nRealizations = DetermineSOWandRealizationNumber()
    if statecuAbortReport == "check":
        CheckForAbortedStateCU()
    else: 
        # create a folder in the scratch directory for analysis if one doesn't exist
        if not os.path.exists(analysisDir):
            os.makedirs("/scratch/ayt5134/analysis/")
            os.system("ln -s /scratch/ayt5134/analysis/ {}".format(analysisDir))

        print("** ANALYZING, BUILDING, PLOTTING STATECU EXPERIMENT ** ")  
        StateCUDataReader.ReadCDS().to_csv(analysisDir + r"/CropInfo.csv", index=False)
        StateCUDataReader.ReadIPY().to_csv(analysisDir + r"/IrrigationInfo.csv", index=False)
        StateCUDataReader.ReadSTR1().to_csv(analysisDir + r"/ElevationInfo.csv", index=False)
        gsCtrlDF = BuildControlGSIWR()
        gsCtrlDF.to_csv(analysisDir + r"/CtrlGSIWR.csv", index=False)
        os.system("bash {} {} {} {}".format(scriptsDir + r"/AnalysisJobChain.sh", nSOWs, nRealizations, len(set(gsCtrlDF["WDID"].values))))

