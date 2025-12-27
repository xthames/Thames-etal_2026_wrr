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
Input data can be found here: Human, I.M. (2021). My input dataset name [Data set]. DataHub. https://doi.org/some-doi-number

You may also follow the procedure outlined below:
1. Download daily precipitation and temperature observations from [NOAA NCEI](https://www.ncei.noaa.gov/cdo-web/search) for the Upper Colorado River Basin in the state of Colorado. The following are the climate stations used in this experiment, reflecting the key climate stations used from the historic consumptive use analysis [(Garrison, 2015)](https://cdss.colorado.gov/modeling-data/consumptive-use-statecu). Note that some climate stations are missing observations within the period of analysis (1950-2013), and if so nearby secondary stations are used for in-filling.
    | NOAA NCEI Key ID | Key Climate Station Name | Secondary Station ID(s) |
    | --- | --- | --- |
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


### Output data
Reference for each minted data source for your output data.  For example:

Human, I.M. (2021). My output dataset name [Data set]. DataHub. https://doi.org/some-doi-number

_your output data references here_


## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| StateCU | 14.0.1 | https://github.com/OpenCDSS/cdss-app-statecu-fortran | - |

## Reproduce our experiment
1. Install the software components required to conduct the experiment from [contributing modeling software](#contributing-modeling-software)
2. Download and install the supporting [input data](#input-data) required to conduct the experiment
3. Run the following scripts in the `workflow` directory to re-create this experiment:

| Script Name | Description | How to Run |
| --- | --- | --- |
| `step_one.py` | Script to run the first part of my experiment | `python3 step_one.py -f /path/to/inputdata/file_one.csv` |
| `step_two.py` | Script to run the second part of my experiment | `python3 step_two.py -o /path/to/my/outputdir` |

4. Download and unzip the [output data](#output-data) from my experiment 
5. Run the following scripts in the `workflow` directory to compare my outputs to those from the publication

| Script Name | Description | How to Run |
| --- | --- | --- |
| `compare.py` | Script to compare my outputs to the original | `python3 compare.py --orig /path/to/original/data.csv --new /path/to/new/data.csv` |

## Reproduce our figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication.

| Figure Number(s) | Script Name | Description | How to Run |
| --- | --- | --- | --- |
| 1, 2 | `generate_plot.py` | Description of figure, ie. "Plots the difference between our two scenarios" | `python3 generate_plot.py -input /path/to/inputs -output /path/to/outuptdir` |
| 3 | `generate_figure.py` | Description of figure, ie. "Shows how the mean and peak differences are calculated" | `python3 generate_figure.py -input /path/to/inputs -output /path/to/outuptdir` |

