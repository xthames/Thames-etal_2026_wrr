# import
import os
import sys
import numpy as np
import pandas as pd
import StateCUDataReader
from SALib.analyze import delta as SAdelta


# environment variables
nSOWs, nRealizations, wdidIdx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


# filepaths
configsDir = os.path.dirname(os.path.dirname(__file__)) + r"/configs"
controlDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU"
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
analysisDir = os.path.dirname(os.path.dirname(__file__)) + r"/analysis"


# build the changes to IWR, sensitivity analysis, and robustness analysis per user
def BuildIWRChangeSARobustness(): 
    # -- control
    irriUserIdx = irriInfoDF["WDID"] == wdid
    irriUserEntry = irriInfoDF.loc[irriUserIdx]
    ctrlUserIdx = ctrlGSIWRDF["WDID"] == wdid
    ctrlUserEntry = ctrlGSIWRDF.loc[ctrlUserIdx]
    district, elev = list(set(ctrlUserEntry["DISTRICT"].values))[0], list(set(ctrlUserEntry["ELEV"].values))[0]
    ctrlWDIDCrops = sorted(set(ctrlUserEntry["CROP"].values))
    ctrlGSIWRperCropDict = {wdidCrop: {"GS": [], "IWR": []} for wdidCrop in ctrlWDIDCrops}
    wdidIWRs = []
    for year in years:
        ctrlYearIdx = ctrlUserEntry["YEAR"] == year
        ctrlYearEntry = ctrlUserEntry.loc[ctrlYearIdx]
        wdidIWRs.append(np.nansum(ctrlYearEntry["IWR"].values))
        for wdidCrop in ctrlWDIDCrops:
            ctrlGSIWRperCropDict[wdidCrop]["IWR"].append(ctrlYearEntry.loc[ctrlYearEntry["CROP"] == wdidCrop, "IWR"].values[0])
            ctrlGSIWRperCropDict[wdidCrop]["GS"].append(ctrlYearEntry.loc[ctrlYearEntry["CROP"] == wdidCrop, "GS"].values[0])
    # IWR stats
    ctrlAvgIWR = np.nanmean(wdidIWRs)
    ctrlMaxIWR = np.nanmax(wdidIWRs)
    ctrlTotIWR = np.nansum(wdidIWRs)
    # irrigation stats
    ctrlAvgIWRFlood = np.nanmean([wdidIWRs[y]*irriUserEntry.loc[irriUserEntry["YEAR"] == year, "FLOOD FRAC"].values[0] for y, year in enumerate(years)])
    ctrlAvgIWRFlood = np.nan if ctrlAvgIWRFlood == 0. else ctrlAvgIWRFlood
    ctrlAvgIWRSprinkler = np.nanmean([wdidIWRs[y]*irriUserEntry.loc[irriUserEntry["YEAR"] == year, "SPRINKLER FRAC"].values[0] for y, year in enumerate(years)])
    ctrlAvgIWRSprinkler = np.nan if ctrlAvgIWRSprinkler == 0. else ctrlAvgIWRSprinkler
    # crop stats 
    ctrlAvgIWRDict = {wdidCrop: np.nanmean(ctrlGSIWRperCropDict[wdidCrop]["IWR"]) for wdidCrop in ctrlWDIDCrops}
    ctrlAvgIWRDict = {wdidCrop: np.nan if ctrlAvgIWRDict[wdidCrop] == 0 else ctrlAvgIWRDict[wdidCrop] for wdidCrop in ctrlWDIDCrops}
    ctrlAvgGSDict = {wdidCrop: np.nanmean(ctrlGSIWRperCropDict[wdidCrop]["GS"]) for wdidCrop in ctrlWDIDCrops}
    
    # -- experiment
    iwrDict, irriDict, cropDict, robustDict = {}, {}, {}, {}
    expTypes = {"SOW": int, "REALIZATION": int, "WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    expUserEntry = pd.DataFrame()
    for s in range(1, nSOWs+1):
        for r in range(1, nRealizations+1):
            srEntry = pd.read_csv(analysisDir + r"/GSIWR-S{}R{}.csv".format(s, r), dtype=expTypes)
            srUserIdx = srEntry["WDID"] == wdid
            srUserEntry = srEntry.loc[srUserIdx]
            expUserEntry = srUserEntry if expUserEntry.empty else pd.concat([expUserEntry, srUserEntry])
    expUserEntry.reset_index(drop=True, inplace=True) 
    expWDIDCrops = sorted(set(expUserEntry["CROP"].values))
    sowTotChangeRaws, sowTotChangeNorms = [], []
    for s in sorted(set(expUserEntry["SOW"].values)):
        sowIdx = expUserEntry["SOW"] == s
        rzTotChangeRaws, rzTotChangeNorms = [], []
        for r in sorted(set(expUserEntry["REALIZATION"].values)):
            rzIdx = expUserEntry["REALIZATION"] == r
            expRzEntry = expUserEntry.loc[sowIdx & rzIdx]
            totalIWRRaw, totalIWRNorm = 0., 0.
            aboveMaxRaw, aboveMaxNorm = 0., 0.
            for year in years:
                irriYearIdx = irriUserEntry["YEAR"] == year
                irriYearEntry = irriUserEntry.loc[irriYearIdx]
                expYearIdx = expRzEntry["YEAR"] == year
                expYearEntry = expRzEntry.loc[expYearIdx]
                # IWR
                expRawIWR, expNormIWR = 0., 0.
                for wdidCrop in expWDIDCrops:
                    expCropGS = expYearEntry.loc[expYearEntry["CROP"] == wdidCrop, "GS"].values[0]
                    expCropIWR = expYearEntry.loc[expYearEntry["CROP"] == wdidCrop, "IWR"].values[0]
                    expCropIWR = 0. if expCropIWR < 0 else expCropIWR
                    expRawIWR += expCropIWR
                    expCropNormIWR = expCropIWR * (ctrlAvgGSDict[wdidCrop] / expCropGS)
                    cropDict[(wdid, s, r, year, wdidCrop)] = [wdid, district, elev, s, r, year, wdidCrop, 
                                                              round(expCropIWR, 3), round(expCropNormIWR, 3), 
                                                              round(100. * (expCropIWR - ctrlAvgIWRDict[wdidCrop]) / ctrlAvgIWRDict[wdidCrop], 3), 
                                                              round(100. * (expCropNormIWR - ctrlAvgIWRDict[wdidCrop]) / ctrlAvgIWRDict[wdidCrop], 3)]
                    expNormIWR += expCropNormIWR
                # totaling, change
                totalIWRRaw += expRawIWR
                totalIWRNorm += expNormIWR
                rawChange = round(100. * (expRawIWR - ctrlAvgIWR) / ctrlAvgIWR, 3)
                normChange = round(100. * (expNormIWR - ctrlAvgIWR) / ctrlAvgIWR, 3)
                iwrDict[(wdid, s, r, year)] = [wdid, district, elev, s, r, year, rawChange, normChange]
                # irrigation method
                expRawIWRFlood = expRawIWR * irriYearEntry["FLOOD FRAC"].values[0] 
                expRawIWRSprinkler = expRawIWR * irriYearEntry["SPRINKLER FRAC"].values[0]
                expNormIWRFlood = expNormIWR * irriYearEntry["FLOOD FRAC"].values[0]
                expNormIWRSprinkler = expNormIWR * irriYearEntry["SPRINKLER FRAC"].values[0]
                rawFloodChange = round(100. * (expRawIWRFlood - ctrlAvgIWRFlood) / ctrlAvgIWRFlood, 3)
                rawSprinklerChange = round(100. * (expRawIWRSprinkler - ctrlAvgIWRSprinkler) / ctrlAvgIWRSprinkler, 3)
                normFloodChange = round(100. * (expNormIWRFlood - ctrlAvgIWRFlood) / ctrlAvgIWRFlood, 3)
                normSprinklerChange = round(100. * (expNormIWRSprinkler - ctrlAvgIWRSprinkler) / ctrlAvgIWRSprinkler, 3)
                irriDict[(wdid, s, r, year)] = [wdid, district, elev, s, r, year, rawFloodChange, rawSprinklerChange, normFloodChange, normSprinklerChange]
                # robust
                aboveMaxRaw += 1. if expRawIWR > ctrlMaxIWR else 0.
                aboveMaxNorm += 1. if expNormIWR > ctrlMaxIWR else 0.
            # total change
            rzTotChangeRaws.append(round(100. * (totalIWRRaw - ctrlTotIWR) / ctrlTotIWR, 3))
            rzTotChangeNorms.append(round(100. * (totalIWRNorm - ctrlTotIWR) / ctrlTotIWR, 3))
            # robust
            aboveMaxRaw *= 100. / len(years)
            aboveMaxNorm *= 100. / len(years)
            robustDict[(wdid, s, r)] = [wdid, district, elev, s, r, aboveMaxRaw, aboveMaxNorm]
        # average realization total change per sow
        sowTotChangeRaws.append(np.nanmean(rzTotChangeRaws))
        sowTotChangeNorms.append(np.nanmean(rzTotChangeNorms))
    
    # IWR CHANGE
    iwrDF = pd.DataFrame().from_dict(iwrDict, orient="index", columns=["WDID", "DISTRICT", "ELEV", "SOW", "REALIZATION", "YEAR", 
                                                                       "RAW CHANGE", "NORM CHANGE"])
    iwrDF.reset_index(drop=True, inplace=True) 
    iwrDF.astype({"WDID": str, "DISTRICT": int, "ELEV": float, "SOW": int, "REALIZATION": int, "YEAR": int, 
                  "RAW CHANGE": float, "NORM CHANGE": float})
    iwrDF.to_csv(analysisDir + r"/IWRChange-{}.csv".format(wdid), index=False)

    # IRRIGATION CHANGE
    irriDF = pd.DataFrame().from_dict(irriDict, orient="index", columns=["WDID", "DISTRICT", "ELEV", "SOW", "REALIZATION", "YEAR", 
                                                                         "RAW FLOOD CHANGE", "RAW SPRINKLER CHANGE", "NORM FLOOD CHANGE", 
                                                                         "NORM SPRINKLER CHANGE"])
    irriDF.reset_index(drop=True, inplace=True) 
    irriDF.astype({"WDID": str, "DISTRICT": int, "ELEV": float, "SOW": int, "REALIZATION": int, "YEAR": int, 
                   "RAW FLOOD CHANGE": float, "RAW SPRINKLER CHANGE": float, "NORM FLOOD CHANGE": float, "NORM SPRINKLER CHANGE": float})
    irriDF.to_csv(analysisDir + r"/IWRIrrigationChange-{}.csv".format(wdid), index=False) 

    # PER CROP IWR CHANGE
    cropDF = pd.DataFrame().from_dict(cropDict, orient="index", columns=["WDID", "DISTRICT", "ELEV", "SOW", "REALIZATION", "YEAR", 
                                                                         "CROP", "RAW IWR", "NORM IWR", "RAW CHANGE", "NORM CHANGE"])
    cropDF.reset_index(drop=True, inplace=True) 
    cropDF.astype({"WDID": str, "DISTRICT": int, "ELEV": float, "SOW": int, "REALIZATION": int, "YEAR": int, "CROP": str,  
                   "RAW IWR": float, "NORM IWR": float, "RAW CHANGE": float, "NORM CHANGE": float})
    cropDF.to_csv(analysisDir + r"/IWRCrops-{}.csv".format(wdid), index=False)

    # ROBUST
    robustDF = pd.DataFrame().from_dict(robustDict, orient="index", columns=["WDID", "DISTRICT", "ELEV", "SOW", "REALIZATION", "RAW", "NORM"])
    robustDF.reset_index(drop=True, inplace=True) 
    robustDF.astype({"WDID": str, "DISTRICT": int, "ELEV": float, "SOW": int, "REALIZATION": int, "RAW": float, "NORM": float})
    robustDF.to_csv(analysisDir + r"/IWRRobustness-{}.csv".format(wdid), index=False)
    
    # SA
    saResultsRaw = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeRaws), print_to_console=False, num_resamples=10)
    saResultsNorm = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeNorms), print_to_console=False, num_resamples=10)
    np.save(analysisDir + r"/SAUserRaw-{}.npy".format(wdid), saResultsRaw) 
    np.save(analysisDir + r"/SAUserNorm-{}.npy".format(wdid), saResultsNorm) 
        

if __name__ == "__main__":
    # load in
    # -- control GS/IWR
    ctrlTypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    ctrlGSIWRDF = pd.read_csv(analysisDir + r"/CtrlGSIWR.csv", dtype=ctrlTypes)
    # -- irrigation
    irriInfoTypes = {"WDID": str, "YEAR": int, "AREA": float, "FLOOD FRAC": float, "SPRINKLER FRAC": float}
    irriInfoDF = pd.read_csv(analysisDir + r"/IrrigationInfo.csv")
    irriInfoDF.astype(irriInfoTypes)
    # -- sows problem for SA
    sowDict = np.load(syntheticDir + r"/CMIP6/SOTWs.npy", allow_pickle=True).item()
    sowParams = sowDict["sows"]
    bounds = [[np.min(sowParams[:, j]), np.max(sowParams[:, j])] for j in range(sowParams.shape[1])]
    deltaProblem = {"num_vars": sowParams.shape[1],
                    "names": ["mean_precip", "std_precip", "copula_hp", "mean_temp", "std_temp"],
                    "bounds": bounds}
    
    # wdids, crops
    wdids, years, crops = sorted(set(ctrlGSIWRDF["WDID"].values)), sorted(set(ctrlGSIWRDF["YEAR"].values)), sorted(set(ctrlGSIWRDF["CROP"].values))
    wdid = wdids[wdidIdx]
    BuildIWRChangeSARobustness()
