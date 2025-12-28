# import statements
import os
import sys
import pandas as pd
import numpy as np
import CMIP6BiasCorrector
import SynthesizePT


# filepaths
scriptsDir = os.path.dirname(__file__)
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
plotsDir = os.path.dirname(os.path.dirname(__file__)) + r"/plots"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
noaaDir = os.path.dirname(os.path.dirname(__file__)) + r"/noaa"
cmip6Dir = os.path.dirname(os.path.dirname(__file__)) + r"/cmip6"
configDir = os.path.dirname(os.path.dirname(__file__)) + r"/configs"


# identify all of the CMIP6 pathway/model combinations
def GeneratePMCombos():
    pmCounter, pmDict = 0, {}
    # case matching
    match dataToProcess:
        case "NOAA":
            pmCounter += 1
            pmDict[pmCounter] = [pmCounter, r"NOAA"]
        case "CMIP6": 
            sources = [s for s in os.listdir(cmip6Dir) if os.path.isdir(cmip6Dir + r"/{}".format(s))]
            # for each source...
            for source in sources:
                sourceDir = cmip6Dir + r"/{}".format(source)
                pathways = [p for p in os.listdir(sourceDir) if ("ssp" in p) or ("historical" in p)]
                # for each pathway...
                for pathway in pathways:
                    pathwayDir = sourceDir + r"/{}".format(pathway)
                    models = os.listdir(pathwayDir)
                    # for each model...
                    for model in models:
                        modelDir = pathwayDir + r"/{}".format(model)
                        # skip if there isn't any data
                        if len(os.listdir(modelDir)) == 0:
                            continue
                        # update the counter, create the pathway/model string
                        if source == "ornl":
                            for forcing in ["Daymet", "Livneh"]:
                                for downscale in ["DBCCA", "RegCM"]:
                                    pmCounter += 1
                                    pmStr = r"CMIP6/{}/{}/{}/{}/{}".format(source, pathway, model, forcing, downscale)
                                    pmDict[pmCounter] = pmCounter, pmStr
                        else:
                            pmCounter += 1
                            pmStr = r"CMIP6/{}/{}/{}".format(source, pathway, model)
                            pmDict[pmCounter] = [pmCounter, pmStr]
    # create a .txt for each pathway/model combo
    pmDF = pd.DataFrame().from_dict(pmDict, orient="index", columns=["IDX", "DIRPATH"])
    pmDF.reset_index(inplace=True, drop=True)
    pmDF.to_csv(configDir + r"/{}_Config.txt".format(dataToProcess), sep="\t", index=False)
    return pmDF


# directory generation
def CreateDirectories(synth):
    if not synth:
        # plot subdirectories
        plotsSDs = ["gmmhmm", "copulae"]
        
        # for each path in the config file
        for path in pmComboDF["DIRPATH"].values:
            # -- /processed
            if not os.path.exists(processedDir + r"/{}".format(path)):
                os.makedirs(processedDir + r"/{}".format(path))
            else:
                if len(os.listdir(processedDir + r"/{}".format(path))):
                    os.system("rm {}/{}/*".format(processedDir, path))
            
            # -- /plots
            for plotsSD in plotsSDs:
                if not os.path.exists(plotsDir + r"/{}/{}".format(plotsSD, path)):
                    os.makedirs(plotsDir + r"/{}/{}".format(plotsSD, path))
                else:
                    if len(os.listdir(plotsDir + r"/{}/{}".format(plotsSD, path))):
                        os.system("rm {}/{}/{}/*".format(plotsDir, plotsSD, path))

    else:
        # -- /synthetic
        if not os.path.exists(syntheticDir):
            os.makedirs("/scratch/ayt5134/synthetic/")
            os.system("ln -s /scratch/ayt5134/synthetic/{}".format(syntheticDir))
        if not os.path.exists(syntheticDir + r"/{}".format(dataToProcess)):
            os.makedirs(syntheticDir + r"/{}".format(dataToProcess))
        
        # -- remove all the existing scenarios/simulations
        if len(os.listdir(syntheticDir + r"/{}".format(dataToProcess))) > 0:
            os.system("rm -r {}/{}/*".format(syntheticDir, dataToProcess))
        
        # -- /synthetics/path/Scenario#/Sim# 
        for scenario in range(numScenarios):
            for sim in range(numSims):
                os.makedirs(syntheticDir + r"/{}/Scenario{}/Sim{}".format(dataToProcess, scenario+1, sim+1))
        
        # -- /plots
        if not os.path.exists(plotsDir + r"/swg/{}".format(dataToProcess)):
            os.makedirs(plotsDir + r"/swg/{}".format(dataToProcess))
        if len(os.listdir(plotsDir + r"/swg/{}".format(dataToProcess))) > 0:
            os.system("rm -r {}/swg/{}/*".format(plotsDir, dataToProcess))
        if "NOAA" in dataToProcess: 
            for scenario in range(numScenarios):
                os.makedirs(plotsDir + r"/swg/{}/Scenario{}".format(dataToProcess, scenario+1))


if __name__ == "__main__":
    # (0) prune the variables passed to the script, check if we're fitting data or synthesizing 
    match len(sys.argv):
        # too few arguments
        case x if x < 2:
            raise NotImplementedError("You MUST include at least ONE (1) passed environment variable! (noaa | cmip6)")
        
        # enough arguments
        case x if (x == 2) or (x == 4):
            dataToProcess = sys.argv[1].upper()
            if not dataToProcess in ["NOAA", "CMIP6"]:
                raise NotImplementedError("First environment variable MUST be NOAA or CMIP6!")
            synthesize = False
            if len(sys.argv) == 4:
                numScenarios = int(sys.argv[2]) if sys.argv[2].isnumeric() else 0
                if (dataToProcess == "NOAA") and (numScenarios != 1):
                    raise NotImplementedError("NOAA data can only provide a singular (1) SOTW!")
                numSims = int(sys.argv[3]) if sys.argv[2].isnumeric else 0
                synthesize = True
            pmComboDF = GeneratePMCombos() if not os.path.isfile(configDir + r"/{}_Config.txt".format(dataToProcess)) else pd.read_csv(configDir + r"/{}_Config.txt".format(dataToProcess), delimiter="\t")
        
        # anything else
        case _:
            raise NotImplementedError("You MUST include either ONE (1) or THREE (3) passed environment variables, with the last two being integers!")
    
    # if we're fitting the data...
    if not synthesize:
        # create the necessary directories 
        print("** CLEARING EXISTING FIT DATA ** ")
        CreateDirectories(synthesize)
        
        match dataToProcess:
            case "NOAA": 
                print("** FITTING TO AND PLOTTING FROM NOAA UCRB DATASETS **")
                datafitArrayShell = scriptsDir + r"/NOAADataFittingJobArray.sh"
            case "CMIP6":
                print("** FORMULATING CMIP6 PATHWAY/MODEL BIAS CORRECTIONS **")
                CMIP6BiasCorrector.GenerateBiasCorrections()
                print("** FITTING TO AND PLOTTING FROM CMIP6 UCRB DOWNSCALES **")
                datafitArrayShell = scriptsDir + r"/CMIP6DataFittingJobArray.sh"
        
        # submit a job array that steps through each element one at time, submitting a dependent job chain to:
        # -- NOAA
        # (a) read in the historical NOAA UCRB data and bias correct (in parallel, a daily/monthly/bias/complete file for each station)
        # (b) construct the requisite combined datasets (in parallel, removing the per-station files), and construct the GMMHMM/copula fits (in series)
        # (c) plot visualizations, validation for the data, GMMHMM, and copulae (in parallel)
        # -- CMIP6
        # (a) construct the bias-corrected CMIP6 UCRB data per pathway/model (in parallel as a job array)
        # (b) construct the GMMHMM/copula fits (in parallel as a job array)
        # (c) plot visualizations, validation for the data, GMMHMM, and copulae (in parallel, inside a job array)
        os.system("sbatch {}".format(datafitArrayShell))

    if synthesize:
        # create the necessary directories if we're synthesizing data
        print("** CLEARING EXISTING SYNTHESIZED DATA ** ")
        CreateDirectories(synthesize)
        
        # (a) create a joint file with all of the GMMHMM and copulae parameters in it, and sample SOTWs
        # (b) parallelized job for synthesizing precipitation/conditional temperature, writing .prc/.tem/.fd files
        # (c) multithreaded job for plotting everything after the synthesizing has finished
        # (d) process StateCU using each of those SOTWs/realizations
        print("** SYNTHESIZING WEATHER, PLOTTING, AND EXECUTING STATECU ON {} DATA ** ".format(dataToProcess))
        SynthesizePT.GenerateRandomSeeds(dataToProcess, numScenarios, numSims)
        match dataToProcess:
            case "NOAA":
                os.system("sbatch {} {} {} {}".format(scriptsDir + r"/NOAASWGJob.sh", dataToProcess, numScenarios, numSims))
            case "CMIP6":
                os.system("sbatch {} {} {} {}".format(scriptsDir + r"/CMIP6SWGJob.sh", dataToProcess, numScenarios, numSims))

