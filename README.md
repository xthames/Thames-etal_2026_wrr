[![DOI](https://zenodo.org/badge/265254045.svg)](https://zenodo.org/doi/10.5281/zenodo.10442485)

<!-- Get rid of the metarepo instructions (the two sections below this) once you're done. -->

# metarepo
## [Check out the website for instructions](https://immm-sfa.github.io/metarepo)
`metarepo` is short for meta-repository, a GitHub repository that contains instructions to reproduce results in a published work. This repo is a template for creating your own metarepo.

## Purpose
A meta-repository creates a single point of access for someone to find all of the components that were used to create a published work for the purpose of reproducibility. This repository should contain references to all minted data and software as well as any ancillary code used to transform the source data, create figures for your publication, conduct the experiment, and / or execute the contributing software.

<!-- Get rid of the metarepo instructions (the two sections above this) once you're done. -->

# Thames-etal_2026_wrr

**Climate Sensitivity of Agricultural Water Demand Depends on Control Over Growing Season: Implications for Producers in the Upper Colorado River Basin**

Alexander B. Thames<sup>1</sup>, Antonia Hadjimichael<sup>1,2\*</sup>,  and Julianne D. Quinn<sup>3</sup>

<sup>1 </sup>Department of Geosciences, The Pennsylvania State University, University Park, Pennsylvania 16802, USA.

<sup>2 </sup>Earth and Environmental Systems Institute (EESI), The Pennsylvania State University, University Park, Pennsylvania 16802, USA.

<sup>3 </sup>School of Engineering & Applied Science, University of Virginia, Charlottesville, Virginia 22904, USA.

\* corresponding author:  hadjimichael@psu.edu

## Abstract
Impacts of climate change on agricultural water demands are deeply uncertain due to complex interactions between precipitation, temperature, and growing season dynamics. For the different agricultural users experiencing these climate impacts, traditional top-down approaches using downscaled climate projections may underrepresent this uncertainty and mask spatial heterogeneities. To address this shortcoming, we develop a multivariate, multisite, copula-based stochastic weather generator for bottom-up exploratory modeling analysis of agricultural water resources systems. Paired with a regional consumptive use model, this generator allows us to investigate differential impacts of climate change on diverse agricultural producers and crops. We demonstrate this framework in the Upper Colorado River Basin within the state of Colorado. Results show that all producers see irrigation requirements increase higher than their historical averages in more than 50% of our sampled realizations, with producers at lower elevations seeing this increase in more than 75% of them. Global sensitivity analysis reveals that adequate access to water impacts producers' effective growing season lengths and thereby which climate variables most control crop water requirement: producers with adequate water are most sensitive to changes in temperature mean and variance while producers without adequate water are most sensitive to changes in precipitation variance---and not mean---with temperature contributions halving. We also find that access to water is more important than crop type, elevation, or location when considering producer sensitivity to climate. These findings demonstrate how differential vulnerability drivers underscore the need for stakeholder-specific assessments that account for spatial heterogeneity and decision-relevant uncertainties in agricultural water demand.

## Journal reference
Thames, A. B., Hadjimichael, A., & Quinn, J. D. Climate Sensitivity of Agricultural Water Demand Depends on Control Over Growing Season: Implications for Producers in the Upper Colorado River Basin. *Water Resources Research*. (Submitted).

## Data reference

### Input data
Thames, A. (2025). Input Data for Thames et al. -- Climate Sensitivity of Agricultural Water Demand (1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.18071209

You may also follow the procedure outlined below to retrieve and pre-process the input files directly. Note that a prerequisite to this is to establish a directory hierarchy that mirrors what's in the Input data repository and in the [Reproduce our experiment](#reproduce-our-experiment) section. Shell scripting references the SLURM job manager where approriate:
1. Download daily precipitation and temperature observations from [NOAA NCEI](https://www.ncei.noaa.gov/cdo-web/search) for the Upper Colorado River Basin in the state of Colorado. The following are the climate stations used in this experiment, reflecting the key climate stations used from the historic consumptive use analysis [(Garrison, 2015)](https://cdss.colorado.gov/modeling-data/consumptive-use-statecu). Note that some climate stations are missing observations, and if so nearby secondary stations are used for in-filling.
    | NOAA NCEI Key ID | Key Climate Station Name | Secondary Station ID(s) |
    | :---: | :---: | :---: |
    | USC00050214 | Altenbern | USC00056266 |
    | USC00051741 | Collbran | US1COME0133, USC00051743, USC00051745 |
    | USW00023063 | Eagle County AP | US1COEG0006, US1COEG0018, US1COEG0025, USC00057618, USR0000CGYP, USW00003098 |
    | USC00053146 | Fruita 1 W | USC00051772 |
    | USC00053359 | Glenwood Springs #2 | USC00050514 |
    | USC00053489 | Grand Junction 6 ESE | USW00023066 |
    | USC00053500 | Grand Lake | USC00053496 |
    | USC00053592 | Green Mt. Dam | USC00059096 |
    | USC00054664 | Kremmling | USC00053423, USC00054129 |
    | USC00055507 | Meredith | USC00050370, USC00050372, USC00055513 |
    | USC00057031 | Rifle | USC00057033, USW00003016 |
    | USC00059265 | Yampa | USC00056797, USC00057936, USC00059103 |

    After this process is complete, merge data such that only 12 `.csv` files identified by each key climate station exist that contain the key station's and all secondary station's reported observations. The name of the `.csv` files should have the key climate station's name with no numbers, special punctuation, or spaces like:
    | Key Climate Station Name | `.csv` File Name |
    | :---: | :---: |
    | Altenbern | `Altenbern_NOAAClimateDaily.csv` |
    | Collbran | `Collbran_NOAAClimateDaily.csv` |
    | Eagle County AP | `EagleCounty_NOAAClimateDaily.csv` |
    | Fruita 1 W | `Fruita_NOAAClimateDaily.csv` |
    | Glenwood Springs #2 | `GlenwoodSprings_NOAAClimateDaily.csv` |
    | Grand Junction 6 ESE | `GrandJunction_NOAAClimateDaily.csv` |
    | Grand Lake | `GrandLake_NOAAClimateDaily.csv` |
    | Green Mt. Dam | `GreenMtDam_NOAAClimateDaily.csv` |
    | Kremmling | `Kremmling_NOAAClimateDaily.csv` |
    | Meredith | `Meredith_NOAAClimateDaily.csv` |
    | Rifle | `Rifle_NOAAClimateDaily.csv` |
    | Yampa | `Yampa_NOAAClimateDaily.csv` |

2. Download regional daily precipitation and temperature CMIP6 projections from the [NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6)](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6) and the [US Department of Energy SECURE Water Act Assessment (Section 9505)](https://hydrosource.ornl.gov/data/datasets/9505v3flow-1/). The following scripts to assist in this process can be found in `workflow/scripts/`. Note that filepaths must be changed to reflect the local environment. Numerical orderings in the Description section of the table below define how to sequentially run related scripts, if necessary.
    | Script Name | Description | How to Run | Passed Arguments
    | :---: | :---: | :---: | :---: |
    | `DownloadNASACMIP6.py` | Script to (1) create directories and (3) check that all files were successfully downloaded relating to the NASA projections | `python scripts/DownloadNASACMIP6.py` | `makedirs`, `checkdirs` |
    | `DownloadNASACMIP6.sh` | Script to (2) parallelize downloading NASA CMIP6 projections | `sbatch scripts/DownloadNASACMIP6.sh` | - |
    | `DownloadHydroCMIP6.py` | Script to (1) reduce links, (2) create directories, and (4) check that all files were successfully downloaded relating to the DoE/ORNL projections | `python scripts/DownloadHydroCMIP6.py` | `reducelinks`, `createdirs`, `checkdirs` |
    | `DownloadHydroCMIP6.sh` | Script to (3) parallelize downloading Doe/ORNL CMIP6 projections | `sbatch scripts/DownloadHydroCMIP6.sh` | - |

### Output data
Reference for each minted data source for your output data.  For example:

Human, I.M. (2021). My output dataset name [Data set]. DataHub. https://doi.org/some-doi-number

_your output data references here_


## Contributing modeling software
| Model | Version | Repository Link | DOI |
| :---: | :---: | :---: | :---: |
| StateCU | 14.0.1 | https://github.com/OpenCDSS/cdss-app-statecu-fortran | - |

## Reproduce our experiment
1. Create an empty directory and populate it with the hierarchy included in `workflow/`. Both `workflow/configs/` and `workflow/scripts/` are already filled; these contain configuration files or scripts relevant to execute the rest of our experiment
2. Go through all files in `workflow/scripts/` and change any filepaths to reflect your local environment. The scripts included reflect the original Linux/SLURM environment of the first author
3. Populate `workflow/noaa/` with the NOAA observations from the [input data](#input-data). Populate `workflow/cmip6/` with the regional CMIP6 downscales from the [input data](#input-data)   
4. Follow the guidelines (especially the **Development Envionment** section) in the [contributing modeling software](#contributing-modeling-software) to clone the StateCU repository on GitHub to `workflow/cdss-dev/`. After cloning, continue following the instructions to compile the StateCU executable
5. Once StateCU is compiled, change your directory to `cdss-dev/cm2015_StateCU/StateCU/` and create a [symbolic link](https://stackoverflow.com/questions/1951742/how-can-i-symlink-a-file-in-linux) called `statecu_exe` to the executable. Confirm that StateCU is operating nominally by executing it using `./statecu_exe` from the `cdss-dev/cm2015_StateCU/StateCU/` directory and inputting `cm2015B` when prompted. *Note the size of the outputs from running StateCU; make sure you have enough memory to accommodate at least 10,000x this value* 
6. The experiment uses the script and arguments outlined below:
    | Script Name | Description | How to Run |
    | :---: | :---: | :---: |
    | `SWGManager.py` | Script that manages processing of inputs and generation of hydroclimatology via stochastic weather generator | `python scripts/SWGManager.py source numSOWs numRealizations` |
    * `source`: can be `noaa` or `cmip6`. Determines which source of input data to use, the NOAA-based observations or the regional CMIP6 downscales
    * `numSOWs`: optional, can be any positive integer that is the square of a prime number larger than 5<sup>2</sup>. Determines the number of states of the world to consider. If paired with `source=noaa` then it is only permitted that `numSOWs=1`; this experiment considers `numSOWs=1369` in the main text and `numSOWs=121` in the supplement with `source=cmip6`
    * `numRealizations`: optional, can be any positive integer. Determines the number of samples of internal variability for each state of the world. With `source=noaa` we use `numRealizations=1000` and with `source=cmip6` we use `numRealizations=10`

    Explicitly, the following commands were used to run the experiment:
    | Step | Command | Description |
    | :---: | :---: | :---: |
    | I | `python scripts/SWGManager.py noaa` | Process the NOAA observations, extract statistical parameters |
    | II | `python scripts/SWGManager.py noaa 1 1000` | Create stationary, synthetically-generated precipitation, temperature, and frost dates from NOAA observations and run StateCU using them |
    | III | `python scripts/SWGManager.py cmip6` | Process the regional CMIP6 downscaled projections, extract statistical parameters per SSP/model |
    | IV | `python scripts/SWGManager.py cmip6 1369 10` | Create the synthetically-generated precipitation, temperature, and frost dates from the regional CMIP6 downscaled projections and run StateSU using them |
7. Out of the 13,690 realizations processed by StateCU in this way, fewer than 5 fail to be processed. You can check this exact value by running `python scripts/AnalysisManager.py checkdirs`. If any realizations have failed, rerun 6.IV until none do 

## Reproduce our figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication.

| Figure Number(s) | Script Name | Description | How to Run |
| --- | --- | --- | --- |
| 1, 2 | `generate_plot.py` | Description of figure, ie. "Plots the difference between our two scenarios" | `python3 generate_plot.py -input /path/to/inputs -output /path/to/outuptdir` |
| 3 | `generate_figure.py` | Description of figure, ie. "Shows how the mean and peak differences are calculated" | `python3 generate_figure.py -input /path/to/inputs -output /path/to/outuptdir` |

