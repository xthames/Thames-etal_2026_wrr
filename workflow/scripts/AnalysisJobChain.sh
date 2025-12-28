#!/bin/bash

# necessary filepaths
READIN_FILEPATH="/storage/home/ayt5134/work/research/StateCU/scripts/StateCUDataReader.sh"
BUILD_FILEPATH="/storage/home/ayt5134/work/research/StateCU/scripts/BuildAnalysis.sh"
PLOT_FILEPATH="/storage/home/ayt5134/work/research/StateCU/scripts/PlotAnalysis.sh"

# submitting a sequence of jobs that:
# -- reads in the StateCU raw output across all SOWs/realizations
READIN_JOBSTR=$(sbatch ${READIN_FILEPATH} $1 $2)
READIN_JOBID=${READIN_JOBSTR##* }

# -- builds data from the read-in
BUILD_JOBSTR=$(sbatch --dependency=afterok:${READIN_JOBID} ${BUILD_FILEPATH} $1 $2 $3)
BUILD_JOBID=${BUILD_JOBSTR##* }

# -- plots analysis figures 
PLOT_JOBSTR=$(sbatch --dependency=afterok:${BUILD_JOBID} ${PLOT_FILEPATH} $1 $2)
PLOT_JOBID=${PLOT_JOBSTR##* }

