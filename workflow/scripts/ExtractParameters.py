# imports
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from scipy import optimize
from matplotlib import pyplot as plt
import StateCUDataReader


# environment arguments
dataRepo = sys.argv[1]
numScenarios = int(sys.argv[2]) if sys.argv[2].isnumeric() else 0


# filepaths
configsDir = os.path.dirname(os.path.dirname(__file__)) + r"/configs"
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
plotsDir = os.path.dirname(os.path.dirname(__file__)) + r"/plots"
controlDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU"


# load all of the GMMHMM, copulae data into a two dataframes in one dict
def AggregateGMMHMMandCopulaParameters():
    gmmhmmDF, copulaeDF = pd.DataFrame(), pd.DataFrame()
    prcpDF, hpDF, tempDF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    relativeProfilesDict = {param: {sub: {} for sub in ["mean", "std"]} for param in ["PRCP", "HP", "TEMP"]}
    
    # for each filepathway in the config file...
    configPaths = list(configDF["DIRPATH"].values)
    if "CMIP6" in dataRepo:
        # -- add the historical path to the list
        noaaConfigDF = pd.read_csv(configsDir + r"/NOAA_Config.txt", delimiter="\t")
        configPaths.append(noaaConfigDF["DIRPATH"].values[0])
    for path in configPaths:
        # load the files in
        localProcessedDir = processedDir + r"/{}".format(path)
        repoName = path.replace("/", "_")
        if any([p in path for p in skipPathways]) or any([m in path for m in skipModels]): continue
        gmmhmmDict = np.load(localProcessedDir + r"/{}_MultisiteGMMHMMFit.npy".format(repoName), allow_pickle=True).item()
        copulaeDict = np.load(localProcessedDir + r"/{}_CopulaFits.npy".format(repoName), allow_pickle=True).item()
        gmmhmmPathDict, copulaePathDict = {}, {}
        prcpPathDict, hpPathDict, tempPathDict = {}, {}, {}
        
        # for each station...
        meanPs, stdPs = [], []
        for station in stations:
            # ID-ing the GMMHMM stuff
            gmmhmmPathDict[(path, station)] = [path, station, gmmhmmDict[station]["means"][0], gmmhmmDict[station]["stds"][0]] 
            meanPs.append(gmmhmmDict[station]["means"][0])
            stdPs.append(gmmhmmDict[station]["stds"][0])
        relativeProfilesDict["PRCP"]["mean"][path] = [np.nanmean(meanPs), *(np.array(meanPs) / np.nanmean(meanPs))]
        relativeProfilesDict["PRCP"]["std"][path] = [np.nanmean(meanPs), *(np.array(stdPs) / np.nanmean(stdPs))]
        prcpPathDict[path] = [path, np.nanmean(meanPs), np.nanmean(stdPs)]

        # for each month...
        hps = []
        meanTs, stdTs = [], []
        for month in months:
            # ID-ing the copulae stuff
            copulaePathDict[(path, month)] = [path, month, copulaeDict[month]["CopulaDF"].at["Frank", "params"], np.nanmean(copulaeDict[month]["TAVG"]), np.nanstd(copulaeDict[month]["TAVG"])]
            hps.append(copulaeDict[month]["CopulaDF"].at["Frank", "params"])
            meanTs.append(np.nanmean(copulaeDict[month]["TAVG"]))
            stdTs.append(np.nanstd(copulaeDict[month]["TAVG"]))
        relativeProfilesDict["HP"]["mean"][path] = [(np.nanmean(meanPs), np.nanmean(meanTs)), *(np.array(hps) - np.nanmean(hps))] 
        relativeProfilesDict["TEMP"]["mean"][path] = [np.nanmean(meanTs), *(np.array(meanTs) - np.nanmean(meanTs))] 
        relativeProfilesDict["TEMP"]["std"][path] = [np.nanmean(meanTs), *(np.array(stdTs) / np.nanmean(stdTs))] 
        hpPathDict[path] = [path, np.nanmean(hps)]
        tempPathDict[path] = [path, np.nanmean(meanTs), np.nanmean(stdTs)]
        
        # convert these dictionaries to DFs
        # -- to DF
        gmmhmmPathDF = pd.DataFrame().from_dict(gmmhmmPathDict, orient="index", columns=["PATH", "STATION", "MEAN", "STD"])
        copulaePathDF = pd.DataFrame().from_dict(copulaePathDict, orient="index", columns=["PATH", "MONTH", "FRANK", "MEAN", "STD"])
        prcpPathDF = pd.DataFrame().from_dict(prcpPathDict, orient="index", columns=["PATH", "MEAN", "STD"])
        hpPathDF = pd.DataFrame().from_dict(hpPathDict, orient="index", columns=["PATH", "MEAN"])
        tempPathDF = pd.DataFrame().from_dict(tempPathDict, orient="index", columns=["PATH", "MEAN", "STD"])
        # -- reset
        gmmhmmPathDF.reset_index(inplace=True, drop=True), copulaePathDF.reset_index(inplace=True, drop=True), 
        prcpPathDF.reset_index(inplace=True, drop=True), hpPathDF.reset_index(inplace=True, drop=True), tempPathDF.reset_index(inplace=True, drop=True)
        # -- concat
        gmmhmmDF = gmmhmmPathDF if gmmhmmDF is None else pd.concat([gmmhmmDF, gmmhmmPathDF], ignore_index=True)  
        copulaeDF = copulaePathDF if copulaeDF is None else pd.concat([copulaeDF, copulaePathDF], ignore_index=True)  
        prcpDF = prcpPathDF if prcpDF is None else pd.concat([prcpDF, prcpPathDF], ignore_index=True)  
        hpDF = hpPathDF if hpDF is None else pd.concat([hpDF, hpPathDF], ignore_index=True)  
        tempDF = tempPathDF if tempDF is None else pd.concat([tempDF, tempPathDF], ignore_index=True)  
    
    # convert internal relativeProfilDicts to DFs
    for paramk in relativeProfilesDict.keys():
        for statk in relativeProfilesDict[paramk].keys():
            profileCols = ["AVG", *stations] if paramk == "PRCP" else ["AVG", *months]
            relativeProfilesDict[paramk][statk] = pd.DataFrame().from_dict(relativeProfilesDict[paramk][statk], orient="index", columns=profileCols)

    # formatting
    swgAggDict = {"GMMHMM": gmmhmmDF, "Copulae": copulaeDF, "PRCP": prcpDF, "HP": hpDF, "TAVG": tempDF}
    
    # return
    return swgAggDict, relativeProfilesDict 


# sample from all of the parameters simultaneously in a Latin hypercube
def LHCSampling():
    # annually, we have:
    # -- prcp: mean, std
    # -- copula hp: mean
    # -- tavg: mean, std
    # SO: ({prcp annual avg mean} + {prcp annual avg std}) + ({annual avg Frank HP}) + ({temp annual avg mean} + {temp annual avg std}) = 5 parameters total
    # -- organize it like: [PRCP mean, PRCP std, Frank HP, TEMP mean, TEMP std] 
    
    # build the upper and lower bounds, and then -10% of min and +10% of max 
    expandPct, noaaFactor = 0.1, 10.
    lower_bounds, upper_bounds = [], []
    
    # -- prcp
    pMeans = repoParamsDict["PRCP"]["MEAN"].values
    pStds = repoParamsDict["PRCP"]["STD"].values
    pExpandPct = 0.1
    if "CMIP6" in dataRepo:
        lower_bounds.extend([min(pMeans) - abs(expandPct*min(pMeans)), min(pStds) - abs(expandPct*min(pStds))])
        upper_bounds.extend([max(pMeans) + abs(expandPct*max(pMeans)), max(pStds) + abs(expandPct*max(pStds))])
    else:
        lower_bounds.extend([min(pMeans) - abs(noaaFactor*np.finfo(float).eps), min(pStds) - abs(noaaFactor*np.finfo(float).eps)])
        upper_bounds.extend([max(pMeans) + abs(noaaFactor*np.finfo(float).eps), max(pStds) + abs(noaaFactor*np.finfo(float).eps)])
    # -- hp
    hpMeans = repoParamsDict["HP"]["MEAN"].values
    if "CMIP6" in dataRepo:
        lower_bounds.append(min(hpMeans) - abs(expandPct*min(hpMeans)))
        upper_bounds.append(max(hpMeans) + abs(expandPct*max(hpMeans)))
    else:
        lower_bounds.append(min(hpMeans) - abs(noaaFactor*np.finfo(float).eps))
        upper_bounds.append(max(hpMeans) + abs(noaaFactor*np.finfo(float).eps))
    # -- temp
    tMeans = repoParamsDict["TAVG"]["MEAN"].values
    tStds = repoParamsDict["TAVG"]["STD"].values
    if "CMIP6" in dataRepo:
        lower_bounds.extend([min(tMeans) - abs(expandPct*min(tMeans)), min(tStds) - abs(expandPct*min(tStds))])
        upper_bounds.extend([max(tMeans) + abs(expandPct*max(tMeans)), max(tStds) + abs(expandPct*max(tStds))])
    else:
        lower_bounds.extend([min(tMeans) - abs(noaaFactor*np.finfo(float).eps), min(tStds) - abs(noaaFactor*np.finfo(float).eps)])
        upper_bounds.extend([max(tMeans) + abs(noaaFactor*np.finfo(float).eps), max(tStds) + abs(noaaFactor*np.finfo(float).eps)]) 

    # sample from a latin hypercube
    s = 2 if "CMIP6" in dataRepo else 1
    lhcSampler = qmc.LatinHypercube(d=len(upper_bounds), optimization="random-cd", strength=s)
    lhcSample = lhcSampler.random(n=numScenarios)
    lhcSample = qmc.scale(lhcSample, lower_bounds, upper_bounds)
    return lhcSample


# kNN disaggregation for finding the PRCP, copula, TEMP relative profiles
def kNNProfileDisaggregation():
    # (1) create k, weights
    rng = np.random.default_rng()
    k = round(np.sqrt(len(configDF["DIRPATH"].values)))
    w = np.array([(1 / j) for j in range(1, k+1)]) / sum([(1 / j) for j in range(1, k+1)])

    # (2) setup, indexing 
    meanPProfileDF = fullProfilesDict["PRCP"]["mean"]
    pIdxAvgPair = np.reshape([[i, avg] for i, avg in enumerate(meanPProfileDF["AVG"].values)], newshape=(meanPProfileDF.shape[0], 2)) 
    meanTProfileDF = fullProfilesDict["TEMP"]["mean"]
    tIdxAvgPair = np.reshape([[i, avg] for i, avg in enumerate(meanTProfileDF["AVG"].values)], newshape=(meanTProfileDF.shape[0], 2))
    hpProfileDF = fullProfilesDict["HP"]["mean"]
    # -- need to normalize for HP since dimensions for PRCP, TEMP are different
    meanPs, meanTs = meanPProfileDF["AVG"].values, meanTProfileDF["AVG"].values
    if dataRepo == "CMIP6":
        hpIdxAvgPair = [[i, ((avg[0] - np.nanmean(meanPs))/np.nanstd(meanPs), (avg[1] - np.nanmean(meanTs))/np.nanstd(meanTs))] for i, avg in enumerate(hpProfileDF["AVG"].values)]

    # (3) choose a corresponding profile from the SOW parameters 
    profileDict = {i: {prof: [] for prof in ["meanP", "stdP", "meanT", "stdT", "hp"]} for i in range(1, numScenarios+1)}
    for profk in profileDict.keys():
        sow = sows[profk-1]
        sowMeanP, sowStdP = sow[0], sow[1]
        sowMeanT, sowStdT = sow[3], sow[4]

        # sort
        # -- prcp
        diffP = np.reshape([[pIdxAvgPair[i, 0], abs(sowMeanP - pIdxAvgPair[i, 1])] for i in range(pIdxAvgPair.shape[0])], newshape=pIdxAvgPair.shape)
        sortedP = diffP[diffP[:, 1].argsort()]
        pProfileIdx = int(rng.choice(sortedP[:k, 0], p=w)) - 1
        # -- temp
        diffT = np.reshape([[tIdxAvgPair[i, 0], abs(sowMeanT - tIdxAvgPair[i, 1])] for i in range(tIdxAvgPair.shape[0])], newshape=tIdxAvgPair.shape)
        sortedT = diffT[diffT[:, 1].argsort()]
        tProfileIdx = int(rng.choice(sortedT[:k, 0], p=w)) - 1
        # -- hp 
        if dataRepo == "CMIP6":
            sowHPMeanP, sowHPMeanT = (sowMeanP - np.nanmean(meanPs))/np.nanstd(meanPs), (sowMeanT - np.nanmean(meanTs))/np.nanstd(meanTs)
            diffHP = np.reshape([[hpIdxAvgPair[i][0], ((sowHPMeanP - hpIdxAvgPair[i][1][0])**2.) + ((sowHPMeanT - hpIdxAvgPair[i][1][1])**2.)] for i in range(len(hpIdxAvgPair))], 
                                newshape=tIdxAvgPair.shape)
            sortedHP = diffHP[diffHP[:, 1].argsort()]
            hpProfileIdx = int(rng.choice(sortedHP[:k, 0], p=w)) - 1
        else:
            hpProfileIdx = 0

        # make the profiles
        profileDict[profk]["meanP"] = fullProfilesDict["PRCP"]["mean"].iloc[pProfileIdx, 1:].values 
        profileDict[profk]["stdP"] = fullProfilesDict["PRCP"]["std"].iloc[pProfileIdx, 1:].values 
        profileDict[profk]["meanT"] = fullProfilesDict["TEMP"]["mean"].iloc[tProfileIdx, 1:].values 
        profileDict[profk]["stdT"] = fullProfilesDict["TEMP"]["std"].iloc[tProfileIdx, 1:].values 
        profileDict[profk]["hp"] = fullProfilesDict["HP"]["mean"].iloc[hpProfileIdx, 1:].values 

    # return the profile dictionary
    return profileDict


def AggregateCMIP6WX():
    # climate station ID to name dict
    stationDict = {"USC00050214": "Altenbern", "USC00051741": "Collbran",
                   "USW00023063": "Eagle County", "USC00053146": "Fruita",
                   "USC00053359": "Glenwood Springs", "USC00053489": "Grand Junction",
                   "USC00053500": "Grand Lake", "USC00053592": "Green Mt Dam",
                   "USC00054664": "Kremmling", "USC00055507": "Meredith",
                   "USC00057031": "Rifle", "USC00059265": "Yampa"}
    
    # structural climate station weight for each WDID dict
    strDict = StateCUDataReader.ReadSTR2()
    wdids = sorted(list(strDict.keys()))

    # getting CMIP6 projections filepaths
    processedPaths = []
    for stem, dirs, files in os.walk(processedDir):
        for name in files:
            processedPaths.append(os.path.join(stem, name))
    projWXPaths = [path for path in processedPaths if ("NOAA" not in path) and ("historical" not in path) and (not any([True if model in path else False for model in skipModels])) and ("UCRBMonthly.csv" in path)]
    
    # load and save the projection wx
    cmip6WXDict = {}
    for path in projWXPaths:
        projDF = pd.read_csv(path, dtype={"NAME": str, "YEAR": int, "MONTH": str, "PRCP": float, "TAVG": float})
        for wdid in wdids:
            distr = int(wdid[:2])
            stationIDs = list(strDict[wdid].keys())
            prcpWeights = [strDict[wdid][stationID][0] for stationID in stationIDs]
            tempWeights = [strDict[wdid][stationID][1] for stationID in stationIDs]
            for year in sorted(set(projDF["YEAR"].values)): 
                yearIdx = projDF["YEAR"] == year
                yearEntry = projDF.loc[yearIdx]
                prcptot, tempavg = [], []
                for stationID in stationIDs:
                    stationIdx = yearEntry["NAME"] == stationDict[stationID]
                    stationEntry = yearEntry.loc[stationIdx]
                    prcpAgg = np.nan if all(np.isnan(stationEntry["PRCP"].values)) else np.nansum(stationEntry["PRCP"].values)
                    tempAgg = np.nan if all(np.isnan(stationEntry["TAVG"].values)) else np.nanmean(stationEntry["TAVG"].values)
                    prcptot.append(prcpAgg)
                    tempavg.append(tempAgg)
                wdidPRCP = np.nan if any(np.isnan(prcptot)) else 100.*np.average(prcptot, weights=prcpWeights)
                wdidTEMP = np.nan if any(np.isnan(tempavg)) else np.average(tempavg, weights=tempWeights)
                cmip6WXDict[path, wdid, distr, year] = [path, wdid, distr, year, wdidPRCP, wdidTEMP]
    cmip6WXDF = pd.DataFrame().from_dict(cmip6WXDict, orient="index", columns=["PATH", "WDID", "DISTRICT", "YEAR", "PRCP", "TEMP"])
    cmip6WXDF.reset_index(drop=True, inplace=True)
    cmip6WXDF.astype({"PATH": str, "WDID": str, "DISTRICT": int, "YEAR": int, "PRCP": float, "TEMP": float})
    cmip6WXDF.to_csv(controlDir + "/CMIP6_ProjWX.csv", index=False)


# plot annual temperature profile
def PlotParameterProfiles():
    # form the same arrays with the CMIP6 data
    groupLabels = ["NASA SSP126", "NASA SSP245", "NASA SSP370", "NASA SSP585", "ORNL SSP585"] 
    groupColors = ["darkolivegreen", "darkgoldenrod", "darkorange", "darkred", "indianred"] 
    
    # load in the NOAA data
    gmmhmmDict = np.load(processedDir + r"/NOAA/NOAA_MultisiteGMMHMMFit.npy", allow_pickle=True).item()
    copulaeDict = np.load(processedDir + r"/NOAA/NOAA_CopulaFits.npy", allow_pickle=True).item()
    noaaPAVGs, noaaPSTDs, noaaPMINs, noaaPMAXs =[], [], [], []
    for station in stations:
        noaaPMINs.append(min(gmmhmmDict["precipDF"][station].values))    
        noaaPMAXs.append(max(gmmhmmDict["precipDF"][station].values))    
        noaaPAVGs.append(gmmhmmDict[station]["means"][0])
        noaaPSTDs.append(gmmhmmDict[station]["stds"][0])
    noaaHPs = [copulaeDict[month]["CopulaDF"].at["Frank", "params"] for month in months]
    noaaTAVGs, noaaTSTDs, noaaTMINs, noaaTMAXs = [], [], [], []
    for month in months:
        noaaTMINs.append(np.nanmin(copulaeDict[month]["TAVG"]))
        noaaTMAXs.append(np.nanmax(copulaeDict[month]["TAVG"]))
        noaaTAVGs.append(np.nanmean(copulaeDict[month]["TAVG"]))
        noaaTSTDs.append(np.nanstd(copulaeDict[month]["TAVG"]))
    
    # loop over prcp, HP, temp
    for param in ["PRCP", "HP", "TAVG"]:
        if param == "PRCP":
            meanDict = {group: {station: [] for station in stations} for group in groupLabels}
            stdDict = {group: {station: [] for station in stations} for group in groupLabels}
            cmip6DF = cmip6ParamsDict["GMMHMM"]
        else:
            meanDict = {group: {month: [] for month in months} for group in groupLabels}
            stdDict = {group: {month: [] for month in months} for group in groupLabels}
            cmip6DF = cmip6ParamsDict["Copulae"]
        cmip6Paths = [p for p in set(cmip6DF["PATH"].values) if ("historical" not in p) and (p != "NOAA")]
        for path in cmip6Paths:
            cmip6PathIdx = cmip6DF["PATH"] == path
            pathPieces = path.split("/")
            source, pathway, model = pathPieces[1], pathPieces[2], pathPieces[3]
            if param == "HP":
                cmip6PathHPs = cmip6DF.loc[cmip6PathIdx, "FRANK"].values
            else:
                cmip6PathAVGs = cmip6DF.loc[cmip6PathIdx, "MEAN"].values
                cmip6PathSTDs = cmip6DF.loc[cmip6PathIdx, "STD"].values
            if param == "PRCP":
                for s, station in enumerate(stations):
                    cmip6MeanFactor = ((cmip6PathAVGs[s] / np.nanmean(cmip6PathAVGs)) - 1.) * 100.
                    cmip6StdFactor = ((cmip6PathSTDs[s] / np.nanmean(cmip6PathSTDs)) - 1.) * 100.
                    meanDict["{} {}".format(source.upper(), pathway.upper())][station].append(cmip6MeanFactor)
                    stdDict["{} {}".format(source.upper(), pathway.upper())][station].append(cmip6StdFactor)
            else:
                for m, month in enumerate(months):
                    if param == "HP":
                        cmip6MeanShift = cmip6PathHPs[m] - np.nanmean(cmip6PathHPs)
                        meanDict["{} {}".format(source.upper(), pathway.upper())][month].append(cmip6MeanShift)
                    else:
                        cmip6MeanShift = cmip6PathAVGs[m] - np.nanmean(cmip6PathAVGs)
                        cmip6StdFactor = ((cmip6PathSTDs[m] / np.nanmean(cmip6PathSTDs)) - 1.) * 100.
                        meanDict["{} {}".format(source.upper(), pathway.upper())][month].append(cmip6MeanShift)
                        stdDict["{} {}".format(source.upper(), pathway.upper())][month].append(cmip6StdFactor)

        # plots
        if param == "PRCP":
            years = list(gmmhmmDict["precipDF"].index)
            noaaPInterAnnual = ((gmmhmmDict["precipDF"].values.T / np.nanmean(noaaPAVGs)) - 1.) * 100.
            pInterAnnual, axis = plt.subplots(nrows=1, ncols=1, figsize=(14, 9))
            axis.set_title("PRCP Interannual Variability Relative to Spatial Annual Mean")
            axis.set_xlabel("Years"), axis.set_ylabel("Stations")
            axis.set_xticks(np.arange(0, len(years), step=10), years[::10])
            axis.set_yticks(np.arange(0, len(stations), step=3), stations[::3])
            cax = axis.imshow(noaaPInterAnnual, cmap="bwr_r", vmin=-125., vmax=125.)
            cbar = pInterAnnual.colorbar(cax)
            cbar.set_label("Change from Spatial Annual Mean [%]")
            #plt.tight_layout()
            pInterAnnual.savefig(plotsDir + r"/params/{}InterAnnual.svg".format(param))
            plt.close()
        if param == "HP":
            profilePlot, axis = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
        else:
            profilePlot, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9), sharex="all")
        if param == "PRCP":
            profilePlot.supxlabel("Stations")
            x = range(1, len(stations)+1)
        else:
            profilePlot.supxlabel("Months")
            x = range(1, len(months)+1)
        if param == "HP":
            axis.set_ylabel("Frank HP Mean Profile Relative to Annual Mean [-]") 
            for g, group in enumerate(meanDict.keys()):
                mnths = meanDict[group].keys()
                vals = meanDict[group].values()
                monthVals = [np.nanmean(val) for val in vals]
                axis.plot(x, monthVals, color=groupColors[g], marker="o", label=groupLabels[g])
            axis.hlines(0., xmin=0.5, xmax=12.5, colors="black", linestyle="dashed", label=None)
            axis.plot(x, np.array(noaaHPs) - np.nanmean(noaaHPs), color="black", marker="o", label="NOAA")
            axis.legend()
        else:
            for i, axis in enumerate(axes.flat):
                if i == 0:
                    if param == "PRCP":
                        axis.set_ylabel("Precipitation log10(Mean) Profile Relative to Annual log10(Mean) [%]") 
                        axis.fill_between(x, (((np.array(noaaPMINs) / np.nanmean(noaaPAVGs)) - 1.) * 100.), 
                                             (((np.array(noaaPMAXs) / np.nanmean(noaaPAVGs)) - 1.) * 100.), color="royalblue", label="NOAA Min/Max")
                    else:
                        axis.set_ylabel("Temperature Mean Profile Relative to Annual Mean [" + chr(176) + "C]") 
                        axis.fill_between(x, np.array(noaaTMINs) - np.nanmean(noaaTAVGs), np.array(noaaTMAXs) - np.nanmean(noaaTAVGs), color="firebrick", label="NOAA Min/Max")
                    for g, group in enumerate(meanDict.keys()):
                        ks = meanDict[group].keys()
                        vals = meanDict[group].values()
                        monthVals = [np.nanmean(val) for val in vals]
                        axis.plot(x, monthVals, color=groupColors[g], marker="o", label=groupLabels[g])
                    axis.hlines(0., xmin=0.5, xmax=12.5, colors="black", linestyle="dashed", label=None)
                    if param == "PRCP":
                        axis.plot(x, ((np.array(noaaPAVGs) / np.nanmean(noaaPAVGs)) - 1.) * 100., color="black", marker="o", label="NOAA")
                    else:
                        axis.plot(x, np.array(noaaTAVGs) - np.nanmean(noaaTAVGs), color="black", marker="o", label="NOAA")
                    axis.legend() 
                if i == 1:
                    if param == "PRCP":
                        axis.set_ylabel("Precipitation log10(Std) Profile Scale to Annual log10(Std) [%]") 
                    else:
                        axis.set_ylabel("Temperature Std Profile Scale to Annual Std [%]") 
                    for g, group in enumerate(stdDict.keys()):
                        mnths = stdDict[group].keys()
                        vals = stdDict[group].values()
                        monthVals = [np.nanmean(val) for val in vals]
                        axis.plot(x, monthVals, color=groupColors[g], marker="o", label=groupLabels[g])
                    axis.hlines(0., xmin=0.5, xmax=12.5, colors="black", linestyle="dashed", label=None)
                    if param == "PRCP":
                        axis.plot(x, ((np.array(noaaPSTDs) / np.nanmean(noaaPSTDs)) - 1.) * 100., color="black", marker="o", label="NOAA")
                    else:
                        axis.plot(x, ((np.array(noaaTSTDs) / np.nanmean(noaaTSTDs)) - 1.) * 100., color="black", marker="o", label="NOAA")
        # x-axis labling
        if param == "PRCP":
            plt.xticks(np.arange(1, len(stations)+1), stations, rotation=45)
        else:
            plt.xticks(np.arange(1, len(months)+1), months)
        # post-processing 
        plt.tight_layout()
        profilePlot.savefig(plotsDir + r"/params/{}Profile.svg".format(param))
        plt.close()


# plot all of the extracted parameters
def PlotParameters():
    # GMMMC parameters
    def PlotPrcpParams():
        meanDict = {group: {station: [] for station in stations} for group in groupLabels}
        stdDict = {group: {station: [] for station in stations} for group in groupLabels}
        
        # setting up the noaa dictionaries to use for plotting
        noaaGMMHMMDF = noaaParamsDict["GMMHMM"] 
        noaaGMMHMMMeanDict = {station: [] for station in stations}
        noaaGMMHMMStdDict = {station: [] for station in stations}
        for station in stations:
            noaaStationIdx = noaaGMMHMMDF["STATION"] == station
            noaaGMMHMMMeanDict[station].extend(noaaGMMHMMDF.loc[noaaStationIdx, "MEAN"].values)
            noaaGMMHMMStdDict[station].extend(noaaGMMHMMDF.loc[noaaStationIdx, "STD"].values)
        
        # setting up to plot
        cmip6GMMHMMDF = cmip6ParamsDict["GMMHMM"]
        expGMMHMMDF = expParamsDict["GMMHMM"]
        cmip6Paths = [p for p in set(cmip6GMMHMMDF["PATH"].values) if ("historical" not in p) and (p != "NOAA")]
        expSOWs = [i for i in set(expGMMHMMDF["SOW"].values)]
        for station in stations:
            cmip6StationIdx = cmip6GMMHMMDF["STATION"] == station
            expStationIdx = expGMMHMMDF["STATION"] == station
            noaaMean = noaaGMMHMMMeanDict[station]
            noaaStd = noaaGMMHMMStdDict[station]
            for path in cmip6Paths:
                cmip6PathIdx = cmip6GMMHMMDF["PATH"] == path
                pathPieces = path.split("/")
                source, pathway, model = pathPieces[1], pathPieces[2], pathPieces[3]
                # mean
                cmip6Mean = cmip6GMMHMMDF.loc[cmip6PathIdx & cmip6StationIdx, "MEAN"].values[0]
                meanFactor = (np.power(10., cmip6Mean) / np.power(10., noaaMean) - 1.) * 100.
                meanDict["{} {}".format(source.upper(), pathway.upper())][station].append(meanFactor[0])
                # std
                cmip6Std = cmip6GMMHMMDF.loc[cmip6PathIdx & cmip6StationIdx, "STD"].values[0]
                stdFactor = ((cmip6Std / noaaStd) - 1.) * 100.
                stdDict["{} {}".format(source.upper(), pathway.upper())][station].append(stdFactor[0])
            for sow in expSOWs:
                expSOWIdx = expGMMHMMDF["SOW"] == sow
                # mean
                expMean = expGMMHMMDF.loc[expSOWIdx & expStationIdx, "MEAN"].values[0]
                meanFactor = (np.power(10., expMean) / np.power(10., noaaMean) - 1.) * 100.
                meanDict["EXPERIMENT"][station].append(meanFactor[0])
                # std
                expStd = expGMMHMMDF.loc[expSOWIdx & expStationIdx, "STD"].values[0]
                stdFactor = ((expStd / noaaStd) - 1.) * 100.
                stdDict["EXPERIMENT"][station].append(stdFactor[0])
  
        # make box plots for the GMMMC mean parameters
        for pltvar in ["mean", "std"]:
            paramCompPlot, axis = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
            paramCompPlot.supxlabel("Stations")
            boxWidth = 0.1
            if pltvar == "mean":
                paramCompPlot.suptitle("PRCP Annual Mean Parameter Comparison")
                paramCompPlot.supylabel("Annual Mean Shift [%]")
                pltDict = meanDict
                pltText = "Mean"
            else:
                paramCompPlot.suptitle("PRCP Annual Std Parameter Comparison")
                paramCompPlot.supylabel("Annual Std Shift [%]")
                pltDict = stdDict
                pltText = "Std"
            # -- plot boxes
            boxes = []
            for g, group in enumerate(pltDict.keys()):
                stns = pltDict[group].keys()
                vals = pltDict[group].values()
                boxplots = axis.boxplot(vals, sym=None, positions=np.arange(1, len(stns)+1)+(g*boxWidth-(len(pltDict.keys())-1)*boxWidth/2), widths=boxWidth, 
                                        patch_artist=True, boxprops={"facecolor": groupColors[g]}, medianprops={"color": "black"}, flierprops={"marker": ".", "markeredgecolor": groupColors[g]})
                boxes.append(boxplots)
            plt.hlines(0., xmin=0.5, xmax=12.5, colors="black", linestyle="dashed")
            # x-axis labling
            axis.set_xticks(np.arange(1, len(stations)+1))
            axis.set_xticklabels(stations, rotation=45, ha="right")
            axis.tick_params(axis="x", labelsize=len(stations))
            # legend
            axis.legend([box["boxes"][0] for box in boxes], groupLabels, loc="lower right", ncol=len(groupLabels))
            # post-plotting
            plt.tight_layout()
            paramCompPlot.savefig(plotsDir + r"/params/PRCP{}ParameterComparison.svg".format(pltText))
            plt.close()
    
    def PlotTempParams():
        hpDict = {group: {month: [] for month in months} for group in groupLabels}
        meanDict = {group: {month: [] for month in months} for group in groupLabels}
        stdDict = {group: {month: [] for month in months} for group in groupLabels}
        
        # setting up the noaa dictionaries to use for plotting
        noaaCopulaeDF = noaaParamsDict["Copulae"] 
        noaaCopulaeHPDict = {month: [] for month in months}
        noaaCopulaeMEANDict = {month: [] for month in months}
        noaaCopulaeSTDDict = {month: [] for month in months}
        for month in months:
            noaaMonthIdx = noaaCopulaeDF["MONTH"] == month
            noaaCopulaeHPDict[month].extend(noaaCopulaeDF.loc[noaaMonthIdx, "FRANK"].values)
            noaaCopulaeMEANDict[month].extend(noaaCopulaeDF.loc[noaaMonthIdx, "MEAN"].values)
            noaaCopulaeSTDDict[month].extend(noaaCopulaeDF.loc[noaaMonthIdx, "STD"].values)

        # setting up to plot
        cmip6CopulaeDF = cmip6ParamsDict["Copulae"]
        expCopulaeDF = expParamsDict["Copulae"]
        cmip6Paths = [p for p in set(cmip6CopulaeDF["PATH"].values) if ("historical" not in p) and (p != "NOAA")]
        expSOWs = [i for i in set(expCopulaeDF["SOW"].values)]
        for month in months:
            cmip6MonthIdx = cmip6CopulaeDF["MONTH"] == month
            expMonthIdx = expCopulaeDF["MONTH"] == month
            noaaHP = noaaCopulaeHPDict[month]
            noaaTMEAN = noaaCopulaeMEANDict[month][0]
            noaaTSTD = noaaCopulaeSTDDict[month][0]
            for path in cmip6Paths:
                cmip6PathIdx = cmip6CopulaeDF["PATH"] == path
                pathPieces = path.split("/")
                source, pathway, model = pathPieces[1], pathPieces[2], pathPieces[3]
                # hp
                cmip6HP = cmip6CopulaeDF.loc[cmip6PathIdx & cmip6MonthIdx, "FRANK"].values[0]
                hpDict["{} {}".format(source.upper(), pathway.upper())][month].append(cmip6HP)
                # tmean
                cmip6TMEAN = cmip6CopulaeDF.loc[cmip6PathIdx & cmip6MonthIdx, "MEAN"].values[0]
                meanShift = cmip6TMEAN - noaaTMEAN
                meanDict["{} {}".format(source.upper(), pathway.upper())][month].append(meanShift)
                # tstd
                cmip6TSTD = cmip6CopulaeDF.loc[cmip6PathIdx & cmip6MonthIdx, "STD"].values[0]
                stdFactor = ((cmip6TSTD / noaaTSTD) - 1.) * 100.
                stdDict["{} {}".format(source.upper(), pathway.upper())][month].append(stdFactor)
            for sow in expSOWs:
                expSOWIdx = expCopulaeDF["SOW"] == sow
                # hp
                expHP = expCopulaeDF.loc[expSOWIdx & expMonthIdx, "FRANK"].values[0]
                hpDict["EXPERIMENT"][month].append(expHP)
                # tavg
                expTMEAN = expCopulaeDF.loc[expSOWIdx & expMonthIdx, "MEAN"].values[0]
                meanShift = expTMEAN - noaaTMEAN
                meanDict["EXPERIMENT"][month].append(meanShift)
                # tavg
                expTSTD = expCopulaeDF.loc[expSOWIdx & expMonthIdx, "STD"].values[0]
                stdFactor = ((expTSTD / noaaTSTD) - 1.) * 100.
                stdDict["EXPERIMENT"][month].append(stdFactor)

        # make box plots for the GMMMC mean parameters
        for pltvar in ["hp", "tmean", "tstd"]:
            paramCompPlot, axis = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
            paramCompPlot.supxlabel("Months")
            boxWidth = 0.1
            if pltvar == "hp":
                paramCompPlot.suptitle("Copulae Monthly Hyperparameter Comparison")
                paramCompPlot.supylabel("Frank HP [-]")
                pltDict = hpDict
                pltText = "FrankHP"
            elif pltvar == "tmean":
                paramCompPlot.suptitle("TAVG Mean Monthly Comparison")
                paramCompPlot.supylabel("Monthly Temperature Mean Shift [" + chr(176) + "C]")
                pltDict = meanDict
                pltText = "Mean"
                plt.hlines(0., xmin=0.5, xmax=12.5, colors="black", linestyle="dashed")
            else:
                paramCompPlot.suptitle("TAVG Std Monthly Comparison")
                paramCompPlot.supylabel("Monthly Temperature Std Shift [%]")
                pltDict = stdDict
                pltText = "Std"
                plt.hlines(0., xmin=0.5, xmax=12.5, colors="black", linestyle="dashed")
            # -- plot boxes
            boxes = []
            for g, group in enumerate(pltDict.keys()):
                mnths = pltDict[group].keys()
                vals = pltDict[group].values()
                boxplots = axis.boxplot(vals, sym=None, positions=np.arange(1, len(mnths)+1)+(g*boxWidth-(len(pltDict.keys())-1)*boxWidth/2), widths=boxWidth, 
                                        patch_artist=True, boxprops={"facecolor": groupColors[g]}, medianprops={"color": "black"}, flierprops={"marker": ".", "markeredgecolor": groupColors[g]})
                boxes.append(boxplots)
            if pltvar == "hp":
                for m, mnth in enumerate(noaaCopulaeHPDict.keys()):
                    plt.hlines(noaaCopulaeHPDict[mnth], xmin=(m+1.)-0.5, xmax=(m+1.)+0.5, colors="black", linestyle="dashed")
            else:
                plt.hlines(0., xmin=0.5, xmax=12.5, colors="black", linestyle="dashed")

            # x-axis labling
            axis.set_xticks(np.arange(1, len(months)+1))
            axis.set_xticklabels(months)
            axis.tick_params(axis="x", labelsize=len(months))
            # legend
            axis.legend([box["boxes"][0] for box in boxes], groupLabels, loc="lower right", ncol=len(groupLabels))
            # post-plotting
            plt.tight_layout()
            paramCompPlot.savefig(plotsDir + r"/params/TAVG{}ParameterComparison.svg".format(pltText))
            plt.close()
    
    # build the LHS SOWs in the same way as the noaaParams and repoParams
    copulaeDict = np.load(processedDir + r"/NOAA/NOAA_CopulaFits.npy", allow_pickle=True).item()
    expParamsDict = {i: {} for i in ["GMMHMM", "Copulae"]}
    for i in range(1, sows.shape[0]+1):
        # -- pull the selected parameters out from the OLH
        sowParameters = sows[i-1]
        scenarioAnnualPMean, scenarioAnnualPStd = sowParameters[0], sowParameters[1] 
        scenarioAnnualHP = sowParameters[2]
        scenarioAnnualTMean, scenarioAnnualTStd = sowParameters[3], sowParameters[4]

        # -- GMMHMM
        for s, station in enumerate(stations):
            log10ScenarioMean = scenarioAnnualPMean * profilesDict[i]["meanP"][s]
            log10ScenarioStd = scenarioAnnualPStd * profilesDict[i]["stdP"][s]
            expParamsDict["GMMHMM"][(i, station)] = [i, station, log10ScenarioMean, log10ScenarioStd]
        
        # -- Copulae, TAVG
        for m, month in enumerate(months):
            scenarioFrankHP = scenarioAnnualHP + profilesDict[i]["hp"][m]
            scenarioTMean = scenarioAnnualTMean + profilesDict[i]["meanT"][m]
            scenarioTStd = scenarioAnnualTStd * profilesDict[i]["stdT"][m]
            expParamsDict["Copulae"][(i, month)] = [i, month, scenarioFrankHP, scenarioTMean, scenarioTStd]
    
    # convert dictionaries to dataframes 
    for i in ["GMMHMM", "Copulae"]:
        cols = ["SOW", "STATION", "MEAN", "STD"] if i == "GMMHMM" else ["SOW", "MONTH", "FRANK", "MEAN", "STD"]
        expParamsDict[i] = pd.DataFrame().from_dict(expParamsDict[i], orient="index", columns=cols)
        expParamsDict[i].reset_index(drop=True, inplace=True) 

    # groups, group colors
    groupLabels = ["NASA SSP126", "NASA SSP245", "NASA SSP370", "NASA SSP585", "ORNL SSP585", "EXPERIMENT"] 
    groupColors = ["darkolivegreen", "darkgoldenrod", "darkorange", "darkred", "indianred", "grey"]
    
    # plot the params
    PlotPrcpParams()
    PlotTempParams()


# plot all of the chosen profiles
def PlotChosenProfiles(): 
    for var in ["meanP", "stdP", "meanT", "stdT", "hp"]:
        xlabel = stations if var in ["meanP", "stdP"] else months
        fixedline = 0. if var in ["meanT", "hp"] else 1.
        profileCompPlot, axis = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
        profileCompPlot.suptitle("Chosen Experimental Profiles of: {}".format(var.upper()))
        for k in sowDict["profiles"].keys():
            vprof = sowDict["profiles"][k][var]
            vprof = (vprof - 1.) * 100. if var in ["meanP", "stdP", "stdT"] else vprof
            axis.plot([i+1 for i in range(len(vprof))], vprof, color="grey", alpha=0.25)
        axis.hlines(0, 1, len(xlabel), color="black", linestyle="dashed")
        axis.set_xticks(np.arange(1, len(xlabel)+1))
        axis.set_xticklabels(xlabel)
        axis.tick_params(axis="x", labelsize=len(xlabel))
        # post-plotting
        plt.tight_layout()
        profileCompPlot.savefig(plotsDir + r"/params/Chosen{}ProfileComparison.svg".format(var.upper()))
        plt.close()


# running the program
if __name__ == "__main__": 
    # load the file pathways
    configDF = pd.read_csv(configsDir + r"/{}_Config.txt".format(dataRepo), delimiter="\t")
    
    # load in the NOAA observations
    noaaObsDF = pd.read_csv(processedDir + r"/NOAA/NOAA_UCRBMonthly.csv")
    stations = sorted(set(noaaObsDF["NAME"]))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] 
    
    # list the projections, models, stations that misbehave (SSPs have notably different distributions from historical projections for just these stations)
    skipPathways = ["historical"]
    skipModels = ["KACE-1-0-G", "CMCC-CM2-SR5", "TaiESM1"]

    # aggregate from each pathway into a single dict
    repoParamsDict, fullProfilesDict = AggregateGMMHMMandCopulaParameters()
    if dataRepo == "CMIP6":
        np.save(syntheticDir + r"/{}/CMIP6RawParams.npy".format(dataRepo), repoParamsDict) 

    # design the experiment: (orthogonal) Latin Hypercube sampling for our parameters
    sows = LHCSampling()
    profilesDict = kNNProfileDisaggregation() 
    sowDict = {"sows": sows, "profiles": profilesDict} 
    np.save(syntheticDir + r"/{}/SOTWs.npy".format(dataRepo), sowDict)

    # plot that we're adequately covering the parameter space
    if dataRepo == "CMIP6":
        # -- loading, formating
        tempRepo = dataRepo
        dataRepo = "NOAA"
        configDF = pd.read_csv(configsDir + r"/{}_Config.txt".format(dataRepo), delimiter="\t")
        noaaParamsDict, _ = AggregateGMMHMMandCopulaParameters()
        cmip6ParamsDict = repoParamsDict
        dataRepo = tempRepo
        # -- aggregate cmip6 projections, if file doesn't already exist
        if not os.path.isfile(controlDir + "/CMIP6_ProjWX.csv"):
            AggregateCMIP6WX()
        # -- plot
        PlotParameterProfiles() 
        PlotParameters()
        PlotChosenProfiles()

