# imports
import os
import sys
import time
import subprocess


# filepaths
statecuDir = os.path.dirname(os.path.dirname(__file__))
cmip6Dir = statecuDir + r"/cmip6/nasa"


# environmnent variables
models = ["ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CESM2", "CESM2-WACCM",
          "CMCC-CM2-SR5", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "CanESM5",
          "EC-Earth3", "EC-Earth3-Veg-LR", "FGOALS-g3", "GFDL-CM4", "GFDL-CM4_gr2",
          "GFDL-ESM4", "GISS-E2-1-G", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "IITM-ESM",
          "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "KIOST-ESM", 
          "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0",
          "NESM3", "NorESM2-LM", "NorESM2-MM", "TaiESM1", "UKESM1-0-LL"]
pathways = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
weather = ["pr", "tas"]
hist = [str(year) for year in range(1950, 2015)]
forecast = [str(year) for year in range(2015, 2101)]
lats = [str(39.00), str(40.75)]
lons = [str(-109.00), str(-105.25)]
netCDF = "netcdf4-classic"


# make all the necessary directories if they haven't yet been made
def MakeDirectories():
    for pathway in pathways:
        for model in models:
            os.system("mkdir -p {}/{}/{}/".format(cmip6Dir, pathway, model))


# how to use wget to retrieve the data
def wgetCMIP6(pw, mdl, yrs, isMissing=False, pt=None):
    # filepath
    fp = cmip6Dir + "/{}/{}".format(pw, mdl)

    # some conditional statements for the correct urls: GCODE
    if mdl in ["CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3", "EC-Earth3-Veg-LR", "IPSL-CM6A-LR", "KACE-1-0-G"]:
        gcode = "gr"
    elif mdl in ["GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5-0", "KIOST-ESM"]:
        gcode = "gr1"
    elif mdl in ["GFDL-CM4_gr2"]:
        gcode = "gr2"
    else:
        gcode = "gn"
    # some conditional statements for the correct urls: RCODE
    if mdl in ["CESM2"]:
        rcode = "r4i1p1f1"
    elif mdl in ["CESM2-WACCM", "FGOALS-g3"]:
        rcode = "r3i1p1f1"
    elif mdl in ["CNRM-CM6-1", "CNRM-ESM2-1", "GISS-E2-1-G", "MIROC-ES2L", "UKESM1-0-LL"]:
        rcode = "r1i1p1f2"
    elif mdl in ["HadGEM3-GC31-LL", "HadGEM3-GC31-MM"]:
        rcode = "r1i1p1f3"
    else:
        rcode = "r1i1p1f1"
    # some conditional statements for the last day of December (??)
    if mdl in ["HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "KACE-1-0-G", "UKESM1-0-LL"]:
        decLastDay = "30"
    else:
        decLastDay = "31"
    # some conditional statements for just the GFDL-CM4_gr2 model, for whatever reason...
    if mdl == "GFDL-CM4_gr2":
        mdlShort = "GFDL-CM4"
    else:
        mdlShort = mdl

    # looping over the weather variables...
    for wvar in weather:
        if wvar == "pr":
            vcode = "_v1.1"
        else:
            if pw in ["ssp126", "ssp370"] and mdl not in ["CESM2", "CMCC-CM2-SR5", "IITM-ESM"]:
                vcode = "_v1.2"
            elif mdl == "TaiESM1":
                vcode = "_v1.1"
            else:
                vcode = ""

        # for each year in the dataset
        for yr in yrs:
            # skip downloading if the file already exists and is populated with data
            if os.path.isfile("{}/Daily{}{}.nc4".format(fp, wvar.upper(), yr)) and os.path.getsize("{}/Daily{}{}.nc4".format(fp, wvar.upper(), yr)) > 1:
                continue
            # conditional for years that don't exist in that model set
            if mdl == "IITM-ESM" and ((pw in ["ssp126", "ssp245", "ssp585"] and int(yr) == 2100) or (pw == "ssp370" and int(yr) >= 2099)):
                continue
            url = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6"
            url += "/{}/{}/{}/{}/{}_day_{}_{}_{}_{}_{}{}.nc?".format(mdl, pw, rcode, wvar, wvar, mdlShort, pw, rcode, gcode, yr, vcode)
            url += "var={}&north={}&west={}&east={}&south={}&horizStride=1&".format(wvar, lats[1], lons[0], lons[1], lats[0])
            url += "time_start={}-01-01T12:00:00Z&time_end={}-12-{}T12:00:00Z&&&accept={}&addLatLon=true".format(yr, yr, decLastDay, netCDF) 
            wgetStr = "wget --continue --retry-connrefused --tries=100 --waitretry=45 --limit-rate=200k -a {}/output.log -O {}/Daily{}{}.nc4".format(fp, fp, wvar.upper(), yr)
            wgetCmdStr = wgetStr.split(" ")
            subprocess.call([*wgetCmdStr, url])
            time.sleep(150)


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
        case "makedirs":
            MakeDirectories()
        
        case "historical" | "ssp126" | "ssp245" | "ssp370" | "ssp585":
            # get the model number and the years
            modelNum = int(sys.argv[2]) - 1
            years = hist if task == "historical" else forecast
            
            # some models don't have the full set of pathways -- don't try to download anything you know isn't there
            if (task == "ssp370" and models[modelNum] in ["CESM2-WACCM", "GFDL-CM4", "GFDL-CM4_gr2", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "KIOST-ESM", "NESM3"]) or \
               (task == "ssp126" and models[modelNum] in ["CESM2-WACCM", "GFDL-CM4", "GFDL-CM4_gr2"]) or \
               (task == "ssp245" and models[modelNum] in ["HadGEM3-GC31-MM"]):
                pass
            else:
                # download the files!
                wgetCMIP6(task, models[modelNum], years)
        
        case "checkdirs":
            CheckDirectories()
        
        case _:
            raise NotImplementedError("Please pass a valid environment variable!")

