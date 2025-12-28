# import
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from multiprocessing import Process
import StateCUDataReader
import ExtractParameters
import warnings
import cartopy.io.img_tiles as cimgt
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import geopandas as gpd
import pygmt
import matplotlib.patheffects as PathEffects
import scipy.stats as stats
import scipy.interpolate as interpolate
from SALib.analyze import delta as SAdelta
from sklearn.linear_model import LinearRegression


# environment variables
nSOWs, nRealizations = int(sys.argv[1]), int(sys.argv[2])


# filepaths
configsDir = os.path.dirname(os.path.dirname(__file__)) + r"/configs"
controlDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU"
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
analysisDir = os.path.dirname(os.path.dirname(__file__)) + r"/analysis"
plotsDir = os.path.dirname(os.path.dirname(__file__)) + r"/plots"
mapDir = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/MapCU"


# helper function for intrascript parallelization
def MultiprocessAnalysisPlotting(fns, fninputs):
    pross = []
    for i, fn in enumerate(fns):
        p = Process(target=fn, args=fninputs[i])
        p.start()
        pross.append(p)
    for p in pross:
        p.join()


# plot UCRB map with climate stations
def PlotUCRB():
    # read in whatever spatial data we need
    # -- climate station data
    stationDict = {"Altenbern": "USC00050214", "Collbran": "USC00051741",
                   "Eagle County": "USW00023063", "Fruita": "USC00053146",
                   "Glenwood Springs": "USC00053359", "Grand Junction": "USC00053489",
                   "Grand Lake": "USC00053500", "Green Mt Dam": "USC00053592",
                   "Kremmling": "USC00054664", "Meredith": "USC00055507",
                   "Rifle": "USC00057031", "Yampa": "USC00059265"}
    noaaMonthlyWX = pd.read_csv(processedDir + r"/NOAA/NOAA_UCRBMonthly.csv")
    stations = sorted(set(noaaMonthlyWX["NAME"].values))
    stationLatLonDict = {}
    for station in stations:
        stationIdx = noaaMonthlyWX["NAME"] == station
        lat, lon = list(set(noaaMonthlyWX.loc[stationIdx, "LAT"].values))[0], list(set(noaaMonthlyWX.loc[stationIdx, "LON"].values))[0]
        stationLatLonDict[station] = [stationDict[station], lat, lon]
    stationLatLonDF = pd.DataFrame().from_dict(stationLatLonDict, orient="index", columns=["ID", "LAT", "LON"])
    stationLatLonDF.reset_index(drop=True, inplace=True)
    # -- StateMod diversion locations
    structures = pd.read_csv(mapDir + r"/modeled_diversions.csv", index_col=0)
    # -- map extent
    extent = [-109.125, -105.625, 38.875, 40.50]
    # -- background map tiles
    terrain = cimgt.StadiaMapsTiles('terrain-background')
    # -- Basin shape
    shape_feature_large = ShapelyFeature(Reader(mapDir + r"/Water_Districts.shp").geometries(),
                                         ccrs.PlateCarree(), edgecolor='black', linewidth=1.75, facecolor='None')
    shape_feature_small = ShapelyFeature(Reader(mapDir + r"/Water_Districts.shp").geometries(),
                                         ccrs.PlateCarree(), edgecolor='black', linewidth=1, facecolor='None')
    # -- Stream lines
    flow_feature = ShapelyFeature(Reader(mapDir + r"/UCRBstreams.shp").geometries(),
                                  ccrs.PlateCarree(), edgecolor='royalblue', facecolor='None')
    # generate UCRB map
    fig = plt.figure(figsize=(18, 12))
    ax = plt.axes(projection=terrain.crs)
    # -- Set map extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # -- Draw basin
    ax.add_feature(shape_feature_large, facecolor="grey", alpha=0.4)
    # -- Draw streams
    ax.add_feature(flow_feature, alpha=0.8, linewidth=4, zorder=4)
    # -- Draw users
    ax.scatter(structures['X'], structures['Y'], marker = '.', s = 200,
               c ='orange', edgecolors='black', transform=ccrs.PlateCarree(), zorder=4)
    # -- draw climate stations
    ax.scatter(stationLatLonDF["LON"], stationLatLonDF["LAT"], marker = '*', s = 200,
               c ='black', edgecolors='black', transform=ccrs.PlateCarree(), zorder=4)
    for station in stationLatLonDF["ID"].values:
        stationIdx = stationLatLonDF["ID"] == station
        ax.text(stationLatLonDF.loc[stationIdx, "LON"].values[0], stationLatLonDF.loc[stationIdx, "LAT"].values[0] + 0.05, station,
                color='black', size=14, ha='center', va='center', transform=ccrs.PlateCarree(),
                path_effects=[PathEffects.withStroke(linewidth=5, foreground="w", alpha=1)], zorder=10)
    ax.set_axis_off()
    # post-processing
    plt.savefig(plotsDir + r"/analysis/UCRBMap.svg")

    # # UCRB map with elevation
    # os.system("gmt clear sessions")
    # extent = [-109.125, -105.6, 38.875, 40.50]
    # parcels = pd.read_csv(mapDir + r"/modeled_diversions.csv", index_col=0)
    # streams = gpd.read_file(mapDir + r"/UCRBstreams.shp").set_crs(crs="EPSG:4326")
    # elev = pygmt.datasets.load_earth_relief(region=extent, resolution="03s")
    # elevMin, elevMax = np.min(np.array(elev)), np.max(np.array(elev))
    # myCmap = pygmt.makecpt(cmap=["255/183/3,white"], series=["{},{}".format(elevMin, elevMax)], continuous=True, reverse=True)
    # districts = gpd.read_file(mapDir + r"/Water_Districts.shp").set_crs(crs="EPSG:4326")
    # fig = pygmt.Figure()
    # fig.grdimage(grid=elev, projection="M15c", cmap=myCmap, shading=True)
    # fig.colorbar(frame=["a500", "x+lElevation", "y+lm"])
    # fig.plot(data=districts, pen="2p,black", fill=False)
    # fig.plot(data=streams, pen="1p,royalblue", fill=False) 
    # fig.plot(x=parcels["X"], y=parcels["Y"], style="c0.1c", fill="black", pen="0.5p,white")
    # fig.plot(x=stationLatLonDF["LON"], y=stationLatLonDF["LAT"], style="a0.8c", fill="black", pen="0.5p,white")
    # for i, station in enumerate(stationLatLonDF["ID"].values):
    #     stationIdx = stationLatLonDF["ID"] == station
    #     fig.text(text="{}".format(i+1), x=stationLatLonDF.loc[stationIdx, "LON"].values[0], y=stationLatLonDF.loc[stationIdx, "LAT"].values[0],
    #              font="7p,Helvetica-Bold,white")
    # fig.psconvert(prefix=plotsDir + r"/analysis/UCRBMap_Elev", fmt="f")


# plot historical crop fraction per water district as a Marikekko
def PlotCropMarimekko():
    # load historical
    ctrlTypes = {"WDID": str, "DISTRICT": int, "CROP": str, "GS": float, "IWR": float}
    ctrlGSDF = pd.read_csv(analysisDir + r"/CtrlGSIWR.csv",
                           usecols=["WDID", "DISTRICT", "CROP", "GS", "IWR"], 
                           dtype=ctrlTypes)
     
    # plot the Marimekko
    prev = 0.
    cropPlot, axis = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    cropPlot.suptitle("UCRB Crop Marimekko")
    yticks = []
    for distr, elev in ascendingDEs:
        distrIdx = ctrlGSDF["DISTRICT"] == distr
        distrEntry = ctrlGSDF.loc[distrIdx]
        totalIWR = np.nansum(distrEntry["IWR"].values)
        h = 100. * (totalIWR / np.nansum(ctrlGSDF["IWR"].values))
        #h = len(set(distrEntry["WDID"]))
        ytickLocation = prev + h/2.
        yticks.append(ytickLocation)
        otherCropsIdx = np.full(distrEntry.shape[0], fill_value=False) 
        stackedCrops = [0.]
        for crop in list(cropColorDict.keys())[:-1]:
            cropIdx = distrEntry["CROP"] == crop
            otherCropsIdx = otherCropsIdx | cropIdx
            cropEntry = distrEntry.loc[cropIdx]
            stackedCrops.append(100. * (np.nansum(cropEntry["IWR"].values)/totalIWR))
        otherCropsIdx = ~otherCropsIdx
        stackedCrops.append(100. * (np.nansum(distrEntry.loc[otherCropsIdx, "IWR"].values))/totalIWR)
        for n in range(len(stackedCrops)-1):
            bot = np.cumsum(stackedCrops[:n+1])[-1]
            axis.barh(ytickLocation, stackedCrops[n+1], height=h, left=bot, 
                      color=cropColorDict[list(cropColorDict.keys())[n]], edgecolor="silver", linewidth=0.1)
        prev += h
    # ticks
    axis.set_ylim([0, prev])
    axis.set_yticks(yticks)
    axis.set_yticklabels(["WD{}".format(distr) for distr, elev in ascendingDEs])
    axis.set_xlabel("Fraction of Total District Historical Irrigation Water Requirement Per Crop [%]")
    axis.set_ylabel("Water Districts")
    # post-processing
    plt.tight_layout()
    cropPlot.savefig(plotsDir + r"/analysis/CropMarimekko.svg")
    plt.close()


# construct the experiment design
def BuildExperimentDesign():
    # establish the empty dictionary for each of the repos
    paramsDict = {}
    paramCols = ["REPO", "PMEAN", "PSTD", "HP", "TMEAN", "TSTD"]

    # for each of the different repos 
    for repo in ["historical", "cmip6", "experiment"]:
        # historical
        if repo == "historical":
            sowDict = np.load(syntheticDir + r"/NOAA/SOTWs.npy", allow_pickle=True).item()
            paramsDict[(repo, 1)] = [repo, *sowDict["sows"][0]]
        # experiment
        if repo == "experiment":
            sowDict = np.load(syntheticDir + r"/CMIP6/SOTWs.npy", allow_pickle=True).item()
            for i in range(sowDict["sows"].shape[0]):
                paramsDict[(repo, i+1)] = [repo, *sowDict["sows"][i]]
        # cmip6
        if repo == "cmip6":
            rawCMIP6Dict = np.load(syntheticDir + r"/CMIP6/CMIP6RawParams.npy", allow_pickle=True).item()
            rawPRCPDF, rawHPDF, rawTEMPDF = rawCMIP6Dict["PRCP"], rawCMIP6Dict["HP"], rawCMIP6Dict["TAVG"] 
            paths = rawPRCPDF["PATH"].values 
            for i, path in enumerate(paths):
                prcpIdx = rawPRCPDF["PATH"] == path
                hpIdx = rawHPDF["PATH"] == path
                tempIdx = rawTEMPDF["PATH"] == path
                rawRepo = "nasa" if "nasa" in path else "ornl"
                paramsDict[(rawRepo, i+1)] = [rawRepo, 
                                              rawPRCPDF.loc[prcpIdx, "MEAN"].values[0], rawPRCPDF.loc[prcpIdx, "STD"].values[0], 
                                              rawHPDF.loc[hpIdx, "MEAN"].values[0],
                                              rawTEMPDF.loc[tempIdx, "MEAN"].values[0], rawTEMPDF.loc[tempIdx, "STD"].values[0]]

    # convert to a dataframe
    paramsDF = pd.DataFrame().from_dict(paramsDict, orient="index", columns=paramCols)

    # save the DF
    paramsDF.to_csv(analysisDir + r"/ExperimentDesign.csv", index=False)


# plot the experiment design
def PlotExperimentDesign():
    # load in the experiment parameters
    designDF = pd.read_csv(analysisDir + r"/ExperimentDesign.csv")
    paramsDF = designDF[designDF.columns[1:]].copy()
    paramsDF["PMEAN"] = 100. * np.power(10., paramsDF["PMEAN"].values)

    # design masks
    maskColors = ["grey", "darkgoldenrod", "blueviolet", "black"]
    maskSizes = [5, 5, 5, 20]
    histMask = designDF["REPO"] == "historical"
    nasaMask = designDF["REPO"] == "nasa"
    ornlMask = designDF["REPO"] == "ornl"
    expMask = designDF["REPO"] == "experiment"

    # make a dataframe filled with the scaled/shifted values
    relParamsDF = paramsDF.copy()
    for col in relParamsDF.columns:
        if col in ["PMEAN", "PSTD"]:
            relParamsDF[col] = relParamsDF[col].values / relParamsDF.loc[histMask, col].values[0]
        else:
            relParamsDF[col] = relParamsDF[col].values - relParamsDF.loc[histMask, col].values[0]

    # axis names
    paramLabels = [r"$\mu_{\mathrm{p}}$", 
                   r"$\sigma_{\mathrm{p}}$", 
                   r"$\theta$", 
                   r"$\mu_{\mathrm{T}}$", 
                   r"$\sigma_{\mathrm{T}}$"]
    
    # design plots
    designPlot, axes = plt.subplots(nrows=relParamsDF.shape[1], ncols=relParamsDF.shape[1], figsize=(14, 9))
    for a, axis in enumerate(axes.flat):
        # index which axis
        i, j = a // relParamsDF.shape[1], a % relParamsDF.shape[1]
        if i <= j:
          axis.set_axis_off()
          continue
        # turn off all the axes
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        # add specific axes, labels
        if i == paramsDF.shape[1]-1: 
            axis.set_xlabel(paramLabels[j])
            axis.xaxis.set_visible(True)
        if j == 0:
            axis.set_ylabel(paramLabels[i])
            axis.yaxis.set_visible(True)
        # plot
        for m, mask in enumerate([expMask, nasaMask, ornlMask, histMask]):
            axis.scatter(relParamsDF.values[mask, j], relParamsDF.values[mask, i], c=maskColors[m], s=maskSizes[m])
    designPlot.savefig(plotsDir + r"/analysis/ExperimentDesign.svg")
    plt.close()


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


# plot the change in WX variables relative to parcel elevation
def PlotUserWXDistributionChange():
    # load colormap info
    colorImg = mpl.image.imread(mapDir + r"/GMT_dem1.png")
    terrainCmap = mpl.colors.LinearSegmentedColormap.from_list("terrain", colorImg[0, :, :])

    # load elev, control wx data
    wdids = sorted(set(elevDF["WDID"].values)) 
    controlWXDF = StateCUDataReader.ReadWXs("control")
    controlWXDtypes = {col: float for col in controlWXDF}
    controlWXDtypes["WDID"], controlWXDtypes["YEAR"] = str, int
    controlWXDF.astype(controlWXDtypes)
    controlWXDF = controlWXDF[((controlWXDF["YEAR"] >= 1950) & (controlWXDF["YEAR"] <= 2013))]
    ctrlDistricts = []
    for i in range(controlWXDF.shape[0]):
        rowEntry = controlWXDF.iloc[i]
        ctrlDistricts.append(int(rowEntry["WDID"][:2]))
    controlWXDF["DISTRICT"] = ctrlDistricts

    # bin the annual precip/temp data
    wxDF, chunkDF, chunksize, counter = pd.DataFrame(), pd.DataFrame(), int(np.sqrt(nSOWs * nRealizations)), 0
    for s in range(1, nSOWs+1):
        for r in range(1, nRealizations+1):
            counter += 1
            srEntry = pd.read_csv(analysisDir + r"/WXChange-S{}R{}.csv".format(s, r), 
                                  usecols=["WDID", "DISTRICT", "ELEV", "PRCP", "TEMP"],
                                  dtype={"WDID": str, "DISTRICT": int, "ELEV": float, "PRCP": float, "TEMP": float})
            if counter >= chunksize:
                wxDF = chunkDF if wxDF.empty else pd.concat([wxDF, chunkDF])
                counter, chunkDF = 0, pd.DataFrame()
            else:
                chunkDF = srEntry if chunkDF.empty else pd.concat([chunkDF, srEntry])
    
    # plot change in precip/temp by elevation/district as boxes
    wxPlot, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    wxPlot.suptitle("Change in Precip, Temp Relative to Historical Average")
    # -- boxplots
    boxWidth = 0.25
    districts = [d for d, e in ascendingDEs] 
    elevs = [e for d, e in ascendingDEs]
    for i, axis in enumerate(axes.flat):
        if i == 0:
            axis.set_xlabel(r"Change in Precipitation Relative to Hist. Average [%]")
            axis.set_ylabel("Water District")
            wcolor, mcolor = mpl.colors.to_rgba("royalblue", alpha=0.5), "black"
            ctrlCol, dataCol = "PRCTOT", "PRCP"
        if i == 1:
            axis.set_xlabel(r"Change in Temperature Relative to Hist. Average [" + chr(176) + "C]")
            wcolor, mcolor = mpl.colors.to_rgba("firebrick", alpha=0.5), "black"
            ctrlCol, dataCol = "TEMAVG", "TEMP"
        for d, distr in enumerate(districts):
            ctrlEntry = controlWXDF.loc[controlWXDF["DISTRICT"] == distr]
            userEntry = wxDF.loc[wxDF["DISTRICT"] == distr]
            histAvg = np.nanmean(100.*ctrlEntry[ctrlCol].values) if i == 0 else np.nanmean(ctrlEntry[ctrlCol].values)
            ctrl = 100. * ((100.*ctrlEntry[ctrlCol].values - histAvg) / histAvg) if i == 0 else ctrlEntry[ctrlCol] - histAvg
            exp = 100. * ((userEntry[dataCol].values - histAvg) / histAvg) if i == 0 else userEntry[dataCol] - histAvg
            cdistrictBox = {"whislo": np.nanpercentile(ctrl, 0), 
                            "q1": np.nanpercentile(ctrl, 25), 
                            "med": np.nanpercentile(ctrl, 50),
                            "q3": np.nanpercentile(ctrl, 75),
                            "whishi": np.nanpercentile(ctrl, 100)}
            districtBox = {"whislo": np.nanpercentile(exp, 0), 
                            "q1": np.nanpercentile(exp, 25), 
                            "med": np.nanpercentile(exp, 50),
                            "q3": np.nanpercentile(exp, 75),
                            "whishi": np.nanpercentile(exp, 100)}
            axis.bxp([cdistrictBox], showfliers=False, positions=[d-0.5*boxWidth], widths=[boxWidth], zorder=10, vert=False, 
                     patch_artist=True, boxprops={"facecolor": wcolor}, medianprops={"color": mcolor})
            axis.bxp([districtBox], showfliers=False, positions=[d+0.5*boxWidth], widths=[boxWidth], zorder=10, vert=False, 
                     patch_artist=True, boxprops={"facecolor": "grey"}, medianprops={"color": mcolor})
        axis.vlines(0, -boxWidth, (len(districts)-1)+boxWidth, linestyle="dashed", colors="black", zorder=12)
        if i == 0:
            axis.set_yticks([d for d in range(len(districts))])
            axis.set_yticklabels(["WD{}".format(d) for d in districts])
        if i == 1:
            axis.set_yticks([])
        axis.set_ylim([-2*boxWidth, (len(districts)-1)+2*boxWidth])    
    # post-processing
    plt.subplots_adjust(bottom=0.175)
    caxis1 = wxPlot.add_axes([0.15, 0.05, 0.7, 0.016])
    cbar = mpl.colorbar.ColorbarBase(caxis1, cmap=terrainCmap, norm=mpl.colors.Normalize(vmin=1.35, vmax=4.25), orientation="horizontal", location="bottom", ticklocation="top")
    caxis1.set_xlim([1.4, 2.45])
    caxis2 = caxis1.twiny()
    caxis2.xaxis.set_ticks_position("bottom")
    caxis2.xaxis.set_label_position("bottom")
    caxis2.set_xlim([1.4, 2.45])
    caxis2.set_xticks([e for e in elevs])
    caxis2.set_xticklabels(["WD{}".format(d) for d in districts], rotation=-30, ha="left")
    caxis1.xaxis.set_ticks_position("top")
    caxis1.xaxis.set_label_position("top")
    cbar.set_label("Elevation [km]")
    #plt.tight_layout()
    wxPlot.savefig(plotsDir + r"/analysis/WXDistChange.svg")
    plt.close()


# plot growing season comparison
def PlotGrowingSeasonComparison(): 
    # load in
    ctrlTypes = {"WDID": str, "CROP": str, "GS": float, "IWR": float}
    ctrlGSDF = pd.read_csv(analysisDir + r"/CtrlGSIWR.csv",
                           usecols=["WDID", "CROP", "GS", "IWR"], 
                           dtype=ctrlTypes)
    wdids = sorted(set(ctrlGSDF["WDID"].values))
    sows = range(1, nSOWs+1) 
    rzs = range(1, nRealizations+1) 
     
    # count fraction of basin-wide IWR for each crop, rank descending
    ctrlCropIWRs, ctrlTotalIWR = [], np.nansum(ctrlGSDF["IWR"].values)
    for crop in sorted(set(cropDict.values())):
        cropCtrlIdx = ctrlGSDF["CROP"] == crop
        cropCtrlEntry = ctrlGSDF.loc[cropCtrlIdx]
        cropTotalIWR = np.nansum(cropCtrlEntry["IWR"].values)
        ctrlCropIWRs.append((crop, round(100.*cropTotalIWR/ctrlTotalIWR, 1)))
    descendingCropIWRs = sorted(ctrlCropIWRs, key=lambda x: x[1], reverse=True) 
    
    # count num of basin-wide parcels that grow the crop, rank descending
    ctrlCropNParcels = []
    for crop in sorted(set(cropDict.values())):
        numParcels = 0
        cropCtrlIdx = ctrlGSDF["CROP"] == crop
        cropCtrlEntry = ctrlGSDF.loc[cropCtrlIdx]
        ctrlCropNParcels.append((crop, len(set(cropCtrlEntry["WDID"].values))))
    descendingCropNParcels = sorted(ctrlCropNParcels, key=lambda x: x[1], reverse=True)

    # threading all SR GS plots together
    expTypes = {"SOW": int, "REALIZATION": int, "WDID": str, "CROP": str, "GS": float}
    expGSDF, chunkDF, chunksize, counter = pd.DataFrame(), pd.DataFrame(), int(np.sqrt(nSOWs * nRealizations)), 0
    for s in sows:
        for r in rzs:
            counter += 1
            srEntry = pd.read_csv(analysisDir + r"/GSIWR-S{}R{}.csv".format(s, r),
                                  usecols=["SOW", "REALIZATION", "WDID", "CROP", "GS"],
                                  dtype=expTypes)
            if counter >= chunksize:
                expGSDF = chunkDF if expGSDF.empty else pd.concat([expGSDF, chunkDF])
                counter, chunkDF = 0, pd.DataFrame()
            else:
                chunkDF = srEntry if chunkDF.empty else pd.concat([chunkDF, srEntry])
    del chunkDF
    # setting up plot
    gsPlot = plt.figure(figsize=(16, 9))
    gsPlot.suptitle("Shift to Length of Growing Season Distribution")
    gsPlot.supxlabel("Length of Growing Season [days]"), gsPlot.supylabel("Cumulative Probability [-]")
    grid = mpl.gridspec.GridSpec(3, 6, figure=gsPlot)
    # developing CDFs
    crops = ["Full UCRB", *[c for c, iwr in descendingCropIWRs]]
    for c, crop in enumerate(crops):
        # reduce by crop
        if crop != "Full UCRB":
            cropCtrlIdx, cropExpIdx = ctrlGSDF["CROP"] == crop, expGSDF["CROP"] == crop
            cropCtrlEntry, cropExpEntry = ctrlGSDF.loc[cropCtrlIdx], expGSDF.loc[cropExpIdx]
        else:
            cropCtrlEntry, cropExpEntry = ctrlGSDF, expGSDF
        # dfs for exp cdfs
        gsPctlDict = {}
        for s in sows:
            sowIdx = cropExpEntry["SOW"] == s
            for r in rzs:
                rzIdx = cropExpEntry["REALIZATION"] == r
                growingSeasonLengths = cropExpEntry.loc[sowIdx & rzIdx, "GS"].values
                pctl0 = np.nanpercentile(growingSeasonLengths, 0)
                pctl25 = np.nanpercentile(growingSeasonLengths, 25)
                pctl50 = np.nanpercentile(growingSeasonLengths, 50)
                pctl75 = np.nanpercentile(growingSeasonLengths, 75)
                pctl100 = np.nanpercentile(growingSeasonLengths, 100)
                gsPctlDict[(s, r)] = [s, r, pctl0, pctl25, pctl50, pctl75, pctl100]
        gsPctlDF = pd.DataFrame.from_dict(gsPctlDict, orient="index", columns=["SOW", "REALIZATION", 0, 25, 50, 75, 100])
        gsPctlDF.reset_index(drop=True, inplace=True)

        # create the cumulative distributions
        ctrlECDF = stats.ecdf(cropCtrlEntry["GS"].values[~np.isnan(cropCtrlEntry["GS"].values)])
        exp0ECDF = stats.ecdf(gsPctlDF[0].values[~np.isnan(gsPctlDF[0].values)])
        exp25ECDF = stats.ecdf(gsPctlDF[25].values[~np.isnan(gsPctlDF[25].values)])
        exp50ECDF = stats.ecdf(gsPctlDF[50].values[~np.isnan(gsPctlDF[50].values)])
        exp75ECDF = stats.ecdf(gsPctlDF[75].values[~np.isnan(gsPctlDF[75].values)])
        exp100ECDF = stats.ecdf(gsPctlDF[100].values[~np.isnan(gsPctlDF[100].values)])
    
        # need to interpolate to use fill_betweenx
        expinterp1_0 = interpolate.interp1d(exp0ECDF.cdf.quantiles, exp0ECDF.cdf.probabilities, bounds_error=False, fill_value=(0., 1.)) 
        expinterp1_25 = interpolate.interp1d(exp25ECDF.cdf.quantiles, exp25ECDF.cdf.probabilities, bounds_error=False, fill_value=(0., 1.)) 
        expinterp1_75 = interpolate.interp1d(exp75ECDF.cdf.quantiles, exp75ECDF.cdf.probabilities, bounds_error=False, fill_value=(0., 1.)) 
        expinterp1_100 = interpolate.interp1d(exp100ECDF.cdf.quantiles, exp100ECDF.cdf.probabilities, bounds_error=False, fill_value=(0., 1.)) 
         
        # communal x-axis to use 
        x_0100 = np.linspace(np.nanmin(gsPctlDF[0].values), np.nanmax(gsPctlDF[100].values), 250) 
        x_2575 = np.linspace(np.nanmin(gsPctlDF[25].values), np.nanmax(gsPctlDF[75].values), 250) 

        # defining axes
        if crop != "Full UCRB":
            if crop in cropColorDict:
                cropColor = cropColorDict[crop]
            else:
                cropColor = cropColorDict[list(cropColorDict.keys())[-1]]
            i, j = int((c-1) / 3), ((c-1) % 3) + 3
            axis = gsPlot.add_subplot(grid[i, j])
            if i != 2: axis.set_xticks([])
            axis.yaxis.tick_right()
            if j != 5: 
                axis.set_yticks([])
            titleStr = "{} | {}, {}%".format(crop, descendingCropNParcels[c-1][1], descendingCropIWRs[c-1][1])
        else:
            axis = gsPlot.add_subplot(grid[:, :3])
            cropColor = "grey"
            titleStr = "{} | U={}, IWR=100%".format(crop, len(set(ctrlGSDF["WDID"].values)))
        axis.set_title(titleStr)
        axis.set_xlim([-3, 370]), axis.set_ylim([-0.025, 1.025])
        axis.fill_between(x_0100, expinterp1_100(x_0100), expinterp1_0(x_0100),
                          color=mpl.colors.to_rgba(cropColor, alpha=0.2), edgecolor=mpl.colors.to_rgba(cropColor, alpha=0.2), zorder=10)
        axis.fill_between(x_2575, expinterp1_75(x_2575), expinterp1_25(x_2575),
                          color=mpl.colors.to_rgba(cropColor, alpha=0.3), edgecolor=mpl.colors.to_rgba(cropColor, alpha=0.3), zorder=10)
        axis.plot(exp50ECDF.cdf.quantiles, exp50ECDF.cdf.probabilities, color=cropColor, zorder=12) 
        axis.plot(ctrlECDF.cdf.quantiles, ctrlECDF.cdf.probabilities, color="black", zorder=13) 
    
    # post-processing
    plt.tight_layout()
    gsPlot.savefig(plotsDir + r"/analysis/GrowingSeasonComparison.svg")
    plt.close()


def ConcatNPYUsers():
    saUserRawDict, saUserNormDict = {}, {}
    #if os.path.isfile(analysisDir + r"/SAUserRaw.npy") and os.path.isfile(analysisDir + r"/SAUserNorm.npy"): return
    for wdid in sorted(set(elevDF["WDID"].values)):
        sauserrawfp = analysisDir + r"/SAUserRaw-{}.npy".format(wdid)
        sausernormfp = analysisDir + r"/SAUserNorm-{}.npy".format(wdid)
        try:
            saWDIDRawDict = np.load(sauserrawfp, allow_pickle=True).item()
            saWDIDNormDict = np.load(sausernormfp, allow_pickle=True).item()
            saUserRawDict[wdid] = saWDIDRawDict 
            saUserNormDict[wdid] = saWDIDNormDict 
            os.remove(sauserrawfp), os.remove(sausernormfp)
        except:
            continue
    with open(analysisDir + r"/SAUserRaw.npy", "wb") as f:
        np.save(f, saUserRawDict)
    with open(analysisDir + r"/SAUserNorm.npy", "wb") as f:
        np.save(f, saUserNormDict)


# plot change in IWR with SA
def PlotUserIWRChangeSA():
    # load, format the IWR data
    ctrlTypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    ctrlGSIWRDF = pd.read_csv(analysisDir + r"/CtrlGSIWR.csv", dtype=ctrlTypes)
    wdids = sorted(set(ctrlGSIWRDF["WDID"].values))
    del ctrlGSIWRDF
    plottingKeys = ["RAW CHANGE", "NORM CHANGE"]
    plottingColumns = ["WDID", "ELEV", "DISTRICT", "MIN", "P25", "MEDIAN", "P75", "MAX"]
    plottingDict = {p: {} for p in plottingKeys}
    for wdid in wdids:
        userEntry = pd.read_csv(analysisDir + r"/IWRChange-{}.csv".format(wdid), 
                                usecols=["WDID", "DISTRICT", "ELEV", "RAW CHANGE", "NORM CHANGE"],
                                dtype={"WDID": str, "DISTRICT": int, "ELEV": float, "RAW CHANGE": float, "NORM CHANGE": float})
        distr = list(set(userEntry["DISTRICT"].values))[0]
        elev = list(set(userEntry["ELEV"].values))[0]
        for p in plottingKeys:
            if np.all(np.isnan(userEntry[p].values)):
                userP0, userP25, userP50, userP75, userP100 = np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                userP0, userP50, userP100 = np.nanmin(userEntry[p].values), np.nanmedian(userEntry[p].values), np.nanmax(userEntry[p].values)
                userP25, userP75 = np.nanpercentile(userEntry[p].values, 25), np.nanpercentile(userEntry[p].values, 75)
            plottingDict[p][wdid] = [wdid, elev, distr, userP0, userP25, userP50, userP75, userP100]
    for p in plottingKeys:
        plottingDict[p] = pd.DataFrame().from_dict(plottingDict[p], orient="index", columns=plottingColumns)
        plottingDict[p].reset_index(drop=True, inplace=True)
        plottingDict[p].astype({"WDID": str, "ELEV": float, "DISTRICT": int, "MIN": float, "P25": float, "MEDIAN": float, "P75": float, "MAX": float})
        plottingDict[p].sort_values(by=["ELEV", "MEDIAN"], ascending=[True, False], inplace=True)
    ascendingWDIDs = [plottingDict["RAW CHANGE"]["WDID"].values, plottingDict["NORM CHANGE"]["WDID"].values]

    # load the SA
    saRawDict = np.load(analysisDir + r"/SAUserRaw.npy", allow_pickle=True).item()
    saNormDict = np.load(analysisDir + r"/SAUserNorm.npy", allow_pickle=True).item()
 
    # plot change in IWR by elevation against SA by elevation 
    elevRank = [i+1 for i in range(len(wdids))]
    for j, p in enumerate(plottingKeys):
        if j == 0: saDict = saRawDict
        if j == 1: saDict = saNormDict
        # plot stuff
        iwrsaPlot, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 9), sharey="all")
        iwrsaPlot.suptitle("User IWR ({}) Relative to Historical Average and SA".format(p.title()))
        iwrsaPlot.supylabel("UCRB Users by Increasing Elevation, then Descending Median")
        for i, axis in enumerate(axes.flat):
            # iwr axis
            if i == 0:
                axis.set_xlabel(r"$\Delta$ IWR Relative to Hist. Avg [%]") 
                axis.set_yticks([])
                axis.fill_betweenx(elevRank, plottingDict[p]["MIN"].values, plottingDict[p]["MAX"].values, 
                                   color=mpl.colors.to_rgba(expColor, alpha=0.2), edgecolor=mpl.colors.to_rgba(expColor, alpha=0.2), zorder=10)
                axis.fill_betweenx(elevRank, plottingDict[p]["P25"].values, plottingDict[p]["P75"].values, 
                                   color=mpl.colors.to_rgba(expColor, alpha=0.6), edgecolor=mpl.colors.to_rgba(expColor, alpha=0.6), zorder=11)
                axis.plot(plottingDict[p]["MEDIAN"].values, elevRank, color=(255./255, 183./255, 3./255), linewidth=0.75, zorder=13)
                axis.vlines(0., min(elevRank), max(elevRank), colors="black", linestyles="dashed", zorder=12)
                if p == "NORM CHANGE":
                    axis.set_xlim([-115, 250])
            # sa axis
            if i >= 1:
                if i == 1: 
                    saMethod = "delta"
                    axis.set_xlabel(r"Delta Moment-Independent Index [-]") 
                if i == 2: 
                    saMethod = "S1"
                    axis.set_xlabel(r"Sobol 1$^{\mathrm{st}}$-Order Index [-]") 
                saBands = [[] for _ in range(len(saNames)+1)]
                for wdid in ascendingWDIDs[j]:
                    saBands[0].append(0.)
                    for n in range(len(saNames)):
                        saVal = saDict[wdid][saMethod][n] if wdid in saDict.keys() else 0.
                        saBands[n+1].append(saBands[n][-1] + saVal)
                for n in range(len(saNames)):
                    axis.fill_betweenx(elevRank, saBands[n], saBands[n+1], color=saColors[n], edgecolor="none", zorder=10) 
                axis.set_xlim([0., 1.])
        # -- post-processing
        plt.tight_layout()
        iwrsaPlot.savefig(plotsDir + r"/analysis/IWRDistByElevUser{}_SA.svg".format(p.title().replace(" ", "")))
        plt.close()


# plot how irrigation effects change in IWR 
def PlotIrrigationIWRChangeSA():
    # load
    ctrlTypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    ctrlGSIWRDF = pd.read_csv(analysisDir + r"/CtrlGSIWR.csv", dtype=ctrlTypes)
    wdids = sorted(set(ctrlGSIWRDF["WDID"].values))
    years = sorted(set(ctrlGSIWRDF["YEAR"].values))
    irriInfoTypes = {"WDID": str, "YEAR": int, "AREA": float, "FLOOD FRAC": float, "SPRINKLER FRAC": float}
    irriInfoDF = pd.read_csv(analysisDir + r"/IrrigationInfo.csv", dtype=irriInfoTypes)
 
    # sows problem for SA
    sowDict = np.load(syntheticDir + r"/CMIP6/SOTWs.npy", allow_pickle=True).item()
    sowParams = sowDict["sows"]
    bounds = [[np.min(sowParams[:, j]), np.max(sowParams[:, j])] for j in range(sowParams.shape[1])]
    deltaProblem = {"num_vars": sowParams.shape[1],
                    "names": ["mean_precip", "std_precip", "copula_hp", "mean_temp", "std_temp"],
                    "bounds": bounds}
         
    # SA for flood, sprinkler irrigation methods
    irriTypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "SOW": int, "REALIZATION": int, "YEAR": int, 
                 "RAW FLOOD CHANGE": float, "RAW SPRINKLER CHANGE": float, "NORM FLOOD CHANGE": float, "NORM SPRINKLER CHANGE": float}
    ctrlTotIWRDict = {"Flood": 0., "Sprinkler": 0.}
    # -- transforming from change back to IWR
    irriTotDF, irriTotChunkDF, chunksize, counter = pd.DataFrame(), pd.DataFrame(), int(np.sqrt(len(wdids))), 0
    for wdid in wdids:
        counter += 1
        # -- collecting averages per WDID
        irriInfoUserIdx = irriInfoDF["WDID"] == wdid
        irriInfoUserEntry = irriInfoDF.loc[irriInfoUserIdx]
        ctrlUserIdx = ctrlGSIWRDF["WDID"] == wdid
        ctrlUserEntry = ctrlGSIWRDF.loc[ctrlUserIdx]
        ctrlWDIDIWRs = []
        for year in years:
            ctrlYearIdx = ctrlUserEntry["YEAR"] == year
            ctrlYearEntry = ctrlUserEntry.loc[ctrlYearIdx]
            ctrlWDIDIWRs.append(np.nansum(ctrlYearEntry["IWR"].values))
        ctrlIWRFloods = [ctrlWDIDIWRs[y]*irriInfoUserEntry.loc[irriInfoUserEntry["YEAR"] == year, "FLOOD FRAC"].values[0] for y, year in enumerate(years)]
        ctrlAvgIWRFlood = np.nan if np.nanmean(ctrlIWRFloods) == 0. else np.nanmean(ctrlIWRFloods)
        ctrlTotIWRDict["Flood"] += np.nansum(ctrlIWRFloods)
        ctrlIWRSprinklers = [ctrlWDIDIWRs[y]*irriInfoUserEntry.loc[irriInfoUserEntry["YEAR"] == year, "SPRINKLER FRAC"].values[0] for y, year in enumerate(years)]
        ctrlAvgIWRSprinkler = np.nan if np.nanmean(ctrlIWRSprinklers) == 0. else np.nanmean(ctrlIWRSprinklers)
        ctrlTotIWRDict["Sprinkler"] += np.nansum(ctrlIWRSprinklers)
        # -- applying averages
        irriTotUserEntry = pd.read_csv(analysisDir + r"/IWRIrrigationChange-{}.csv".format(wdid), 
                                       usecols=["SOW", "REALIZATION", "RAW FLOOD CHANGE", "RAW SPRINKLER CHANGE", "NORM FLOOD CHANGE", "NORM SPRINKLER CHANGE"], 
                                       dtype={"SOW": int, "REALIZATION": int, "RAW FLOOD CHANGE": float, "RAW SPRINKLER CHANGE": float,
                                                                              "NORM FLOOD CHANGE": float, "NORM SPRINKLER CHANGE": float})  
        
        irriTotUserEntry["RAW FLOOD CHANGE"] = ctrlAvgIWRFlood * (1. + (irriTotUserEntry["RAW FLOOD CHANGE"].values / 100.))
        irriTotUserEntry["RAW SPRINKLER CHANGE"] = ctrlAvgIWRSprinkler * (1. + (irriTotUserEntry["RAW SPRINKLER CHANGE"].values / 100.))
        irriTotUserEntry["NORM FLOOD CHANGE"] = ctrlAvgIWRFlood * (1. + (irriTotUserEntry["NORM FLOOD CHANGE"].values / 100.))
        irriTotUserEntry["NORM SPRINKLER CHANGE"] = ctrlAvgIWRSprinkler * (1. + (irriTotUserEntry["NORM SPRINKLER CHANGE"].values / 100.))
        irriTotUserEntry.rename(inplace=True, columns={"RAW FLOOD CHANGE": "RAW FLOOD IWR", "RAW SPRINKLER CHANGE": "RAW SPRINKLER IWR",
                                                       "NORM FLOOD CHANGE": "NORM FLOOD IWR", "NORM SPRINKLER CHANGE": "NORM SPRINKLER IWR"})
        if counter >= chunksize:
            irriTotDF = irriTotChunkDF if irriTotDF.empty else pd.concat([irriTotDF, irriTotChunkDF])
            counter, irriTotChunkDF = 0, pd.DataFrame()
        else:
            irriTotChunkDF = irriTotUserEntry if irriTotChunkDF.empty else pd.concat([irriTotChunkDF, irriTotUserEntry])
    del irriTotChunkDF
    # -- aggregating by sow, realization for SA 
    sowTotChangeFloodRaws, sowTotChangeSprinklerRaws = [], []
    sowTotChangeFloodNorms, sowTotChangeSprinklerNorms = [], []
    for s in sorted(set(irriTotDF["SOW"].values)):
        sowIdx = irriTotDF["SOW"] == s
        rzTotChangeFloodRaws, rzTotChangeSprinklerRaws = [], []
        rzTotChangeFloodNorms, rzTotChangeSprinklerNorms = [], []
        for r in sorted(set(irriTotDF["REALIZATION"].values)):
            rzIdx = irriTotDF["REALIZATION"] == r
            irriRzEntry = irriTotDF.loc[sowIdx & rzIdx]
            rzTotChangeFloodRaws.append(100. * ((np.nansum(irriRzEntry["RAW FLOOD IWR"].values) - ctrlTotIWRDict["Flood"]) / ctrlTotIWRDict["Flood"])) 
            rzTotChangeSprinklerRaws.append(100. * ((np.nansum(irriRzEntry["RAW SPRINKLER IWR"].values) - ctrlTotIWRDict["Sprinkler"]) / ctrlTotIWRDict["Sprinkler"])) 
            rzTotChangeFloodNorms.append(100. * ((np.nansum(irriRzEntry["NORM FLOOD IWR"].values) - ctrlTotIWRDict["Flood"]) / ctrlTotIWRDict["Flood"])) 
            rzTotChangeSprinklerNorms.append(100. * ((np.nansum(irriRzEntry["NORM SPRINKLER IWR"].values) - ctrlTotIWRDict["Sprinkler"]) / ctrlTotIWRDict["Sprinkler"])) 
        sowTotChangeFloodRaws.append(np.nanmean(rzTotChangeFloodRaws))
        sowTotChangeSprinklerRaws.append(np.nanmean(rzTotChangeSprinklerRaws))
        sowTotChangeFloodNorms.append(np.nanmean(rzTotChangeFloodNorms))
        sowTotChangeSprinklerNorms.append(np.nanmean(rzTotChangeSprinklerNorms))
    del irriTotDF
    # -- SA
    saResultsFloodRaw = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeFloodRaws), print_to_console=False, num_resamples=10)
    saResultsSprinklerRaw = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeSprinklerRaws), print_to_console=False, num_resamples=10)
    saResultsFloodNorm = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeFloodNorms), print_to_console=False, num_resamples=10)
    saResultsSprinklerNorm = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeSprinklerNorms), print_to_console=False, num_resamples=10)
    saResultsRaw = {"Flood": saResultsFloodRaw, "Sprinkler": saResultsSprinklerRaw}
    saResultsNorm = {"Flood": saResultsFloodNorm, "Sprinkler": saResultsSprinklerNorm}

    # construct the plot
    irriPlot = plt.figure(figsize=(16, 9))
    irriPlot.suptitle("Irrigation Method Distribution and Sensitivity Analysis")
    grid = mpl.gridspec.GridSpec(2, 4, figure=irriPlot)
    boxWidth = 0.25
    floodColor, sprinklerColor = mpl.colors.to_rgb("#776258"), mpl.colors.to_rgb("#AFC2D5")
    # -- distribution axes
    rawDistAxis = irriPlot.add_subplot(grid[:, 0])
    normDistAxis = irriPlot.add_subplot(grid[:, 2])
    for d, distre in enumerate(ascendingDEs):
        distr = distre[0]
        distrEntry = pd.DataFrame()
        for wdid in wdids:
            if int(wdid[:2]) != distr: continue
            irriDF = pd.read_csv(analysisDir + r"/IWRIrrigationChange-{}.csv".format(wdid), 
                                 usecols=["RAW FLOOD CHANGE", "RAW SPRINKLER CHANGE", "NORM FLOOD CHANGE", "NORM SPRINKLER CHANGE"], 
                                 dtype={"RAW FLOOD CHANGE": float, "RAW SPRINKLER CHANGE": float, "NORM FLOOD CHANGE": float, "NORM SPRINKLER CHANGE": float})
            distrEntry = irriDF if distrEntry.empty else pd.concat([distrEntry, irriDF])
        rawFloodChanges = distrEntry["RAW FLOOD CHANGE"].values
        rawSprinklerChanges = distrEntry["RAW SPRINKLER CHANGE"].values
        normFloodChanges = distrEntry["NORM FLOOD CHANGE"].values
        normSprinklerChanges = distrEntry["NORM SPRINKLER CHANGE"].values
        for a, axis in enumerate([rawDistAxis, normDistAxis]):
            if a == 0:
                floodData, sprinklerData = rawFloodChanges, rawSprinklerChanges
                axis.set_title("Raw")
                axis.set_ylabel("Water District ID") 
            if a == 1:
                floodData, sprinklerData = normFloodChanges, normSprinklerChanges
                axis.set_title("Norm")
            if any(~np.isnan(floodData)):
                floodBox = {"whislo": np.nanmin(floodData), 
                            "q1": np.nanpercentile(floodData, 25), 
                            "med": np.nanmedian(floodData),
                            "q3": np.nanpercentile(floodData, 75),
                            "whishi": np.nanmax(floodData)}
                fbx = axis.bxp([floodBox], showfliers=False, positions=[(d+1) - boxWidth/2.], widths=[boxWidth], zorder=10, vert=False,
                               patch_artist=True,
                               boxprops={"facecolor": floodColor}, 
                               medianprops={"color": "black"}, 
                               flierprops={"marker": ".", "markeredgecolor": floodColor})
            if any(~np.isnan(sprinklerData)):
                sprinklerBox = {"whislo": np.nanmin(sprinklerData), 
                                "q1": np.nanpercentile(sprinklerData, 25), 
                                "med": np.nanmedian(sprinklerData),
                                "q3": np.nanpercentile(sprinklerData, 75),
                                "whishi": np.nanmax(sprinklerData)}
                sbx = axis.bxp([sprinklerBox], showfliers=False, positions=[(d+1) + boxWidth/2.], widths=[boxWidth], zorder=10, vert=False,
                               patch_artist=True,
                               boxprops={"facecolor": sprinklerColor}, 
                               medianprops={"color": "black"}, 
                               flierprops={"marker": ".", "markeredgecolor": sprinklerColor})
            axis.vlines(0., 1-boxWidth, len(ascendingDEs)+boxWidth, colors="black", linestyles="dashed", zorder=9)
            axis.set_xlim([-115, 250])
            axis.set_xlabel(r"$\Delta$ IWR from Hist. Avg [%]")
            if a == 0:
                axis.set_yticks([d+1 for d in range(len(ascendingDEs))])
                axis.set_yticklabels(["WD{}".format(distr) for distr, e in ascendingDEs])
            if a == 1:
                axis.set_yticks([])
    # -- SA axes
    rawDeltaAxis, rawS1Axis = irriPlot.add_subplot(grid[0, 1]), irriPlot.add_subplot(grid[1, 1])
    normDeltaAxis, normS1Axis = irriPlot.add_subplot(grid[0, 3]), irriPlot.add_subplot(grid[1, 3])
    saAxes = [rawDeltaAxis, rawS1Axis, normDeltaAxis, normS1Axis]
    saListInfo = [(saD, saM, saAxes[2*i+j]) for i, saD in enumerate([saResultsRaw, saResultsNorm]) for j, saM in enumerate(["delta", "S1"])]
    plotInc = 0
    for saDict, saMethod, axis in saListInfo: 
        plotInc += 1
        axis.set_ylim([0., 1.])
        saBands = [[] for _ in range(len(saNames)+1)]
        for irriMethod in saDict.keys():
            saBands[0].append(0.)
            for n in range(len(saNames)):
                saVal = saDict[irriMethod][saMethod][n] 
                saBands[n+1].append(saVal)
        for n in range(len(saNames)):
            bots = np.cumsum(np.array(saBands)[:n+1, :], axis=0)[-1]
            bars = saBands[n+1]
            axis.bar([i+1 for i in range(len(saDict.keys()))], bars, bottom=bots, color=saColors[n])
        axis.set_xticks([i+1 for i in range(len(saDict.keys()))])
        axis.set_xticklabels(list(saDict.keys()))
        if plotInc == 1: axis.set_title("Raw")
        if (plotInc % 2) == 1: axis.set_xticks([])
        if plotInc <= 2: axis.set_yticks([])
        if plotInc >= 3: 
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
            if plotInc == 3:
                axis.set_title("Norm")
                axis.set_ylabel("Delta Moment-Independent Index [-]")
            else:
                axis.set_ylabel(r"Sobol 1$^{\mathrm{st}}$-Order Index [-]")
    plt.tight_layout()
    irriPlot.savefig(plotsDir + r"/analysis/IWRDistByElevIrrigation_SA.svg")
    plt.close()
    

# plot crop IWR change, SA
def PlotCropIWRChangeSA():
    # load the control, crop IWR
    ctrlTypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    ctrlGSIWRDF = pd.read_csv(analysisDir + r"/CtrlGSIWR.csv", dtype=ctrlTypes)
    wdids = sorted(set(ctrlGSIWRDF["WDID"].values))
    years = sorted(set(ctrlGSIWRDF["YEAR"].values)) 
    ctrlCropIWRs, ctrlTotalIWR = [], np.nansum(ctrlGSIWRDF["IWR"].values)
    for crop in sorted(set(cropDict.values())):
        cropCtrlIdx = ctrlGSIWRDF["CROP"] == crop
        cropCtrlEntry = ctrlGSIWRDF.loc[cropCtrlIdx]
        cropTotalIWR = np.nansum(cropCtrlEntry["IWR"].values)
        ctrlCropIWRs.append((crop, round(100.*cropTotalIWR/ctrlTotalIWR, 1)))
    descendingCropIWRs = sorted(ctrlCropIWRs, key=lambda x: x[1], reverse=True) 
    #crops = sorted(set(ctrlGSIWRDF["CROP"].values))
    crops = [c for c, iwr in descendingCropIWRs]
     
    # sows problem for SA
    sowDict = np.load(syntheticDir + r"/CMIP6/SOTWs.npy", allow_pickle=True).item()
    sowParams = sowDict["sows"]
    bounds = [[np.min(sowParams[:, j]), np.max(sowParams[:, j])] for j in range(sowParams.shape[1])]
    deltaProblem = {"num_vars": sowParams.shape[1],
                    "names": ["mean_precip", "std_precip", "copula_hp", "mean_temp", "std_temp"],
                    "bounds": bounds}

    # building SA, change per crop
    saResultsRaw = {crop: None for crop in crops}
    saResultsNorm = {crop: None for crop in crops}
    for crop in crops:
        ctrlCropIdx = ctrlGSIWRDF["CROP"] == crop
        ctrlCropEntry = ctrlGSIWRDF.loc[ctrlCropIdx]
        cropWDIDs = sorted(set(ctrlCropEntry["WDID"].values))
        ctrlTotCropIWR = np.nansum(ctrlCropEntry["IWR"].values)
        expCropEntry, expCropChunkEntry, chunksize, counter = pd.DataFrame(), pd.DataFrame(), int(np.sqrt(len(cropWDIDs))), 0
        for wdid in cropWDIDs:
            counter += 1
            cropDF = pd.read_csv(analysisDir + r"/IWRCrops-{}.csv".format(wdid), 
                                 usecols=["SOW", "REALIZATION", "CROP", "RAW IWR", "NORM IWR"],
                                 dtype={"SOW": int, "REALIZATION": int, "CROP": str, "RAW IWR": float, "NORM IWR": float})
            expCropIdx = cropDF["CROP"] == crop
            if counter >= chunksize:
                expCropEntry = expCropChunkEntry if expCropEntry.empty else pd.concat([expCropEntry, expCropChunkEntry])
                counter, expCropChunkEntry = 0, pd.DataFrame()
            else:
                expCropChunkEntry = cropDF.loc[expCropIdx] if expCropChunkEntry.empty else pd.concat([expCropChunkEntry, cropDF.loc[expCropIdx]])
        del expCropChunkEntry
        sowTotChangeRaws, sowTotChangeNorms = [], []
        # -- SA
        for s in sorted(set(expCropEntry["SOW"].values)):
            sowIdx = expCropEntry["SOW"] == s
            rzTotChangeRaws, rzTotChangeNorms = [], []
            for r in sorted(set(expCropEntry["REALIZATION"].values)):
                rzIdx = expCropEntry["REALIZATION"] == r
                expRzEntry = expCropEntry.loc[sowIdx & rzIdx]
                rzTotChangeRaws.append(100. * ((np.nansum(expRzEntry["RAW IWR"].values) - ctrlTotCropIWR) / ctrlTotCropIWR))
                rzTotChangeNorms.append(100. * ((np.nansum(expRzEntry["NORM IWR"].values) - ctrlTotCropIWR) / ctrlTotCropIWR))
            sowTotChangeRaws.append(np.nanmean(rzTotChangeRaws))
            sowTotChangeNorms.append(np.nanmean(rzTotChangeNorms))
        saResultsCropRaw = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeRaws), print_to_console=False, num_resamples=10)
        saResultsCropNorm = SAdelta.analyze(deltaProblem, sowParams, np.array(sowTotChangeNorms), print_to_console=False, num_resamples=10)
        saResultsRaw[crop] = saResultsCropRaw
        saResultsNorm[crop] = saResultsCropNorm
    del expCropEntry, cropDF

    # construct the plot
    cropPlot = plt.figure(figsize=(16, 9))
    cropPlot.suptitle("Crop Distribution and Sensitivity Analysis")
    grid = mpl.gridspec.GridSpec(2, 4, figure=cropPlot)
    boxWidth = 0.1
    # -- distribution axes
    rawDistAxis = cropPlot.add_subplot(grid[:, 0])
    normDistAxis = cropPlot.add_subplot(grid[:, 2])
    for d, distre in enumerate(ascendingDEs):
        distr = distre[0]
        distrEntry = pd.DataFrame()
        for wdid in wdids:
            if int(wdid[:2]) != distr: continue
            cropDF = pd.read_csv(analysisDir + r"/IWRCrops-{}.csv".format(wdid), 
                                 usecols=["CROP", "RAW CHANGE", "NORM CHANGE"], 
                                 dtype={"CROP": str, "RAW CHANGE": float, "NORM CHANGE": float})
            distrEntry = cropDF if distrEntry.empty else pd.concat([distrEntry, cropDF])
        rawChanges = distrEntry["RAW CHANGE"].values
        normChanges = distrEntry["NORM CHANGE"].values
        for a, axis in enumerate([rawDistAxis, normDistAxis]):
            if a == 0:
                iwrChanges = rawChanges
                axis.set_title("Raw")
                axis.set_ylabel("Water District ID") 
            if a == 1:
                iwrChanges = normChanges
                axis.set_title("Norm") 
            addCropIdx = np.full(len(rawChanges), False) 
            for c, distrCrop in enumerate(list(cropColorDict.keys())[:-1]):
                distrCropIdx = distrEntry["CROP"] == distrCrop
                addCropIdx = addCropIdx | distrCropIdx
                cropChanges = iwrChanges[distrCropIdx]
                if len(cropChanges) > 0:
                    cropBox = {"whislo": np.nanmin(cropChanges), 
                                "q1": np.nanpercentile(cropChanges, 25), 
                                "med": np.nanmedian(cropChanges),
                                "q3": np.nanpercentile(cropChanges, 75),
                                "whishi": np.nanmax(cropChanges)}
                    fbx = axis.bxp([cropBox], showfliers=False, positions=[(d+1) + (c-5./2)*boxWidth], widths=[boxWidth], zorder=10, vert=False,
                                   patch_artist=True,
                                   boxprops={"facecolor": cropColorDict[distrCrop], "edgecolor": cropColorDict[distrCrop]}, 
                                   medianprops={"color": "black"}, 
                                   whiskerprops={"color": cropColorDict[distrCrop]},
                                   capprops={"color": cropColorDict[distrCrop]},
                                   flierprops={"marker": ".", "markeredgecolor": cropColorDict[distrCrop]})
            addCropIdx = ~addCropIdx
            if sum(addCropIdx) > 0:
                cropChanges = iwrChanges[addCropIdx]
                cropBox = {"whislo": np.nanmin(cropChanges), 
                            "q1": np.nanpercentile(cropChanges, 25), 
                            "med": np.nanmedian(cropChanges),
                            "q3": np.nanpercentile(cropChanges, 75),
                            "whishi": np.nanmax(cropChanges)}
                fbx = axis.bxp([cropBox], showfliers=False, positions=[(d+1) + 5*boxWidth/2.], widths=[boxWidth], zorder=10, vert=False,
                               patch_artist=True,
                               boxprops={"facecolor": cropColorDict[list(cropColorDict.keys())[-1]], 
                                         "edgecolor": cropColorDict[list(cropColorDict.keys())[-1]]}, 
                               medianprops={"color": "black"},
                               whiskerprops={"color": cropColorDict[list(cropColorDict.keys())[-1]]},
                               capprops={"color": cropColorDict[list(cropColorDict.keys())[-1]]},
                               flierprops={"marker": ".", "markeredgecolor": cropColorDict[list(cropColorDict.keys())[-1]]})
            axis.vlines(0., 1-3*boxWidth, len(ascendingDEs)+3*boxWidth, colors="black", linestyles="dashed", zorder=9)
            axis.set_xlim([-115, 300])
            axis.set_xlabel(r"$\Delta$ IWR from Hist. Avg [%]")
            if a == 0:
                axis.set_yticks([d+1 for d in range(len(ascendingDEs))])
                axis.set_yticklabels(["WD{}".format(distr) for distr, e in ascendingDEs])
            if a == 1:
                axis.set_yticks([])
            axis.set_ylim([0, 13])
    # -- SA axes
    rawDeltaAxis, rawS1Axis = cropPlot.add_subplot(grid[0, 1]), cropPlot.add_subplot(grid[1, 1])
    normDeltaAxis, normS1Axis = cropPlot.add_subplot(grid[0, 3]), cropPlot.add_subplot(grid[1, 3])
    saAxes = [rawDeltaAxis, rawS1Axis, normDeltaAxis, normS1Axis]
    saListInfo = [(saD, saM, saAxes[2*i+j]) for i, saD in enumerate([saResultsRaw, saResultsNorm]) for j, saM in enumerate(["delta", "S1"])]
    plotInc = 0
    for saDict, saMethod, axis in saListInfo: 
        plotInc += 1
        axis.set_ylim([0., 1.])
        saBands = [[] for _ in range(len(saNames)+1)]
        for saCrop in saDict.keys():
            saBands[0].append(0.)
            for n in range(len(saNames)):
                saVal = saDict[saCrop][saMethod][n] 
                saBands[n+1].append(saVal)
        for n in range(len(saNames)):
            bots = np.cumsum(np.array(saBands)[:n+1, :], axis=0)[-1]
            bars = saBands[n+1]
            axis.bar([i+1 for i in range(len(saDict.keys()))], bars, bottom=bots, color=saColors[n])
        axis.set_xticks([i+1 for i in range(len(saDict.keys()))])
        axis.set_xticklabels(list(saDict.keys()), rotation=-45, ha="left")
        if plotInc == 1: axis.set_title("Raw")
        if (plotInc % 2) == 1: axis.set_xticks([])
        if plotInc <= 2: axis.set_yticks([])
        if plotInc >= 3: 
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
            if plotInc == 3: 
                axis.set_title("Norm")
                axis.set_ylabel("Delta Moment-Independent Index [-]")
            else:
                axis.set_ylabel(r"Sobol 1$^{\mathrm{st}}$-Order Index [-]")
    # -- post-processing
    plt.tight_layout()
    cropPlot.savefig(plotsDir + r"/analysis/IWRDistByElevCrop_SA.svg")
    plt.close()


# robustness/satisficing criteria plots
def PlotUserRobustness():
    # setup 
    robustDtypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "SOW": int, "REALIZATION": int, "RAW": float, "NORM": float}
    ctrlTypes = {"WDID": str, "DISTRICT": int, "ELEV": float, "YEAR": int, "CROP": str, "GS": float, "IWR": float}
    ctrlGSIWRDF = pd.read_csv(analysisDir + r"/CtrlGSIWR.csv", dtype=ctrlTypes)
    ctrlTotIWR = np.nansum(ctrlGSIWRDF["IWR"].values)
    districts = sorted(set(ctrlGSIWRDF["DISTRICT"].values))
    wdids = sorted(set(ctrlGSIWRDF["WDID"].values))
    iwrsToPlot = ["RAW", "NORM"] 
    percentageOfYearsMaxExceeded = np.linspace(10., 100., 10, True)
    percentageOfDistrictUsers = np.linspace(10., 100., 10, True)

    # sort the districts -- ascending elevation
    ascendingDEUFs = []
    for d, e in ascendingDEs:
        districtCtrlIWRIdx = ctrlGSIWRDF["DISTRICT"] == d
        nUsers = len(set(ctrlGSIWRDF.loc[districtCtrlIWRIdx, "WDID"].values))
        distrIWRFrac = round(100. * np.nansum(ctrlGSIWRDF.loc[districtCtrlIWRIdx, "IWR"].values)/ctrlTotIWR, 2)
        ascendingDEUFs.append((d, e, nUsers, distrIWRFrac))
    descendingDEUFs = ascendingDEUFs[::-1]

    # plotting robustness
    robustPlot, axes = plt.subplots(nrows=len(districts), ncols=2, figsize=(9, 16), sharex="all", sharey="all")
    robustPlot.suptitle("Robustness Performance Thresholds per District, Normalization")
    robustPlot.supxlabel("Percent of Users in District [%]")
    robustPlot.supylabel("Percent of Years in Realization where Historical Maximum Exceeded [%]")
    satisficingCmap = "PuOr"
    for d in range(len(districts)):
        distr = descendingDEUFs[d][0]
        distrEntry = pd.DataFrame()
        for wdid in wdids:
            if int(wdid[:2]) != distr: continue
            robustDF = pd.read_csv(analysisDir + r"/IWRRobustness-{}.csv".format(wdid), dtype=robustDtypes)
            distrEntry = robustDF if distrEntry.empty else pd.concat([distrEntry, robustDF])
        distrEntry.reset_index(drop=True, inplace=True)
        satisficingRawDF = pd.DataFrame(0., index=percentageOfYearsMaxExceeded[::-1], columns=percentageOfDistrictUsers)
        satisficingNormDF = pd.DataFrame(0., index=percentageOfYearsMaxExceeded[::-1], columns=percentageOfDistrictUsers)
        for s in sorted(set(distrEntry["SOW"].values)):
            sowIdx = distrEntry["SOW"] == s
            for r in sorted(set(distrEntry["REALIZATION"].values)):
                rzIdx = distrEntry["REALIZATION"] == r
                rzEntry = distrEntry.loc[sowIdx & rzIdx]
                exceedFreqsRaw = rzEntry["RAW"].values
                exceedFreqsNorm = rzEntry["NORM"].values
                for ePct in percentageOfYearsMaxExceeded:
                    pctUserRaw = 100. * np.nansum(exceedFreqsRaw >= ePct) / len(exceedFreqsRaw)
                    for uPct in percentageOfDistrictUsers:
                        if pctUserRaw <= uPct:
                            satisficingRawDF.at[ePct, uPct] += 1
                for ePct in percentageOfYearsMaxExceeded:
                    pctUserNorm = 100. * np.nansum(exceedFreqsNorm >= ePct) / len(exceedFreqsNorm)
                    for uPct in percentageOfDistrictUsers:
                        if pctUserNorm <= uPct:
                            satisficingNormDF.at[ePct, uPct] += 1
        satisficingRawDF *= 100. / (max(robustDF["SOW"].values) * max(robustDF["REALIZATION"].values))
        satisficingNormDF *= 100. / (max(robustDF["SOW"].values) * max(robustDF["REALIZATION"].values))
        rawAxis, normAxis = axes[d, 0], axes[d, 1]
        rawAxis.yaxis.set_label_position("right")
        rawAxis.yaxis.set_label_coords(1.175, 0.5)
        rawAxis.set_ylabel("WD{}".format(distr), rotation=-90)
        normAxis.yaxis.set_label_position("right")
        normAxis.yaxis.set_label_coords(1.175, 0.5)
        normAxis.set_ylabel("{} | {}%".format(descendingDEUFs[d][2], descendingDEUFs[d][3]), rotation=-90)
        if d == 0:
            rawAxis.set_title("Raw")
            normAxis.set_title("Norm")
        if d == 11:
            myticks = [0, 4, 9]
            rawAxis.set_xticks(myticks)
            rawAxis.set_xticklabels([int(list(satisficingRawDF.columns)[t]) for t in myticks])
            rawAxis.set_yticks(myticks)
            rawAxis.set_yticklabels([int(list(satisficingRawDF.index)[t]) for t in myticks])
        rawIm = rawAxis.imshow(satisficingRawDF.values, cmap=satisficingCmap, vmin=0, vmax=100)
        normAxis.imshow(satisficingNormDF.values, cmap=satisficingCmap, vmin=0, vmax=100)
    robustPlot.subplots_adjust(bottom=0.125)
    cbar_ax = robustPlot.add_axes([0.15, 0.075, 0.7, 0.01])
    cbar = robustPlot.colorbar(rawIm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Percent of Realizations where Criteria Satisfied [%]")
    robustPlot.savefig(plotsDir + r"/analysis/RobustnessDistrictMaxExceedance.svg")
    plt.close()
     
    
# run program
if __name__ == "__main__": 
    # establish globals
    AF_to_m3 = 1233.4818375
    expColor = "grey"
    cropDict = BuildCropDict()
    cropColorDict = {"Grass Pasture": "forestgreen", "Alfalfa": "#5E4B56", "Corn Grain": "#E7BF05", 
                     "Spring Grain": "burlywood", "Bluegrass": "cornflowerblue", 
                     "Bluegrass\nCovered Orchard\nUncovered Orchard\nGrapes\nVegetables": "plum"} 
    saNames = ["mean_precip", "std_precip", "copula_hp", "mean_temp", "std_temp"]
    saColors = [mpl.colors.to_rgba("midnightblue", alpha=0.6),
                mpl.colors.to_rgba("midnightblue", alpha=0.2),
                mpl.colors.to_rgba("darkmagenta", alpha=0.4),
                mpl.colors.to_rgba("firebrick", alpha=0.6),
                mpl.colors.to_rgba("firebrick", alpha=0.2)]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] 

    # districts and their average 
    elevDF = pd.read_csv(analysisDir + r"/ElevationInfo.csv", dtype={"WDID": str, "ELEV": float})
    ascendingElevDict = {}
    for wdid in sorted(set(elevDF["WDID"].values)):
        wdidIdx = elevDF["WDID"] == wdid
        wdidDistrict = wdid[:2]
        wdidElev = np.nanmean(elevDF.loc[wdidIdx, "ELEV"].values / 1000.)
        if wdidDistrict not in ascendingElevDict:
            ascendingElevDict[wdidDistrict] = [wdidElev]
        else:
            ascendingElevDict[wdidDistrict].append(wdidElev)
    des = [(int(k), round(np.nanmean(v), 3)) for k, v in ascendingElevDict.items()]
    ascendingDEs = sorted(des, key=lambda x: x[1])

    # build data
    BuildExperimentDesign()
    ConcatNPYUsers()

    # make plots
    #PlotUCRB()
    #PlotCropMarimekko()
    #PlotExperimentDesign()
    #PlotUserWXDistributionChange()
    #PlotGrowingSeasonComparison()
    #PlotUserIWRChangeSA() 
    #PlotIrrigationIWRChangeSA()
    #PlotCropIWRChangeSA()
    #PlotUserRobustness()
    plotFns = [PlotUCRB, PlotCropMarimekko, PlotExperimentDesign, PlotUserWXDistributionChange, PlotGrowingSeasonComparison,
               PlotUserIWRChangeSA, PlotIrrigationIWRChangeSA, PlotCropIWRChangeSA, PlotUserRobustness]
    MultiprocessAnalysisPlotting(plotFns, [()]*len(plotFns))

