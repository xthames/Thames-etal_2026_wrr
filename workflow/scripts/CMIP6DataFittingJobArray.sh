#!/bin/bash
#SBATCH --mail-user=ayt5134@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=azh5924_b
#SBATCH --partition=sla-prio
#SBATCH --job-name=statecu_CMIP6DataFittingJobArray
#SBATCH --output=/storage/home/ayt5134/work/research/StateCU/jobs/output/%x_%a.out
#SBATCH --error=/storage/home/ayt5134/work/research/StateCU/jobs/error/%x_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --array=1-212%39


# (0a) establishing which job this array element will run
CONFIG_FILEPATH="/storage/home/ayt5134/work/research/StateCU/configs/CMIP6_Config.txt"
# -- awk to identify which path to use (with a local variable, match the job array index with corresponding row's dirpath)
DATA_PATH=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $CONFIG_FILEPATH)


# (0b) load in the appropriate modules, venvs
module load anaconda
conda activate statecu_env
module load parallel


# (1) bias correct the CMIP6 observations
PY_STR_BIASCORR="python /storage/home/ayt5134/work/research/StateCU/scripts/CMIP6BiasCorrector.py"
SRUN_OPTS_BIASCORR="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=1"
PARALLEL_OPTS_BIASCORR="--joblog /storage/home/ayt5134/work/research/StateCU/jobs/parallel/statecu_CMIP6BiasCorrect.log --delay 0.1auto --jobs 1 -N 1"
srun $SRUN_OPTS_BIASCORR $PY_STR_BIASCORR $DATA_PATH &
wait


# (2) fit the NOAA observations with a GMMHMM and copulae
PY_STR_FIT="python /storage/home/ayt5134/work/research/StateCU/scripts/GMMHMMandCopulaConstructor.py" 
SRUN_OPTS_FIT="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=1"
srun $SRUN_OPTS_FIT $PY_STR_FIT $DATA_PATH &
wait


# (3) validate the fits
PY_STR_VALIDATE="python /storage/home/ayt5134/work/research/StateCU/scripts/PlotGMMHMMandCopulae.py"
SRUN_OPTS_VALIDATE="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=5"
srun $SRUN_OPTS_VALIDATE $PY_STR_VALIDATE $DATA_PATH &
wait
