#!/bin/bash
#SBATCH --mail-user=ayt5134@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=azh5924_cr_default
#SBATCH --partition=standard
#SBATCH --job-name=statecu_AnalysisPlot
#SBATCH --output=/storage/home/ayt5134/work/research/StateCU/jobs/output/%x.out
#SBATCH --error=/storage/home/ayt5134/work/research/StateCU/jobs/error/%x.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48


# (0) load in the appropriate modules, venvs
module load anaconda
conda activate statecu_env
module load parallel


# (1) plot SWG-generated StateCU data for the analysis
PY_STR_PLOT="python /storage/home/ayt5134/work/research/StateCU/scripts/PlotAnalysis.py"
SRUN_OPTS_PLOT="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
srun $SRUN_OPTS_PLOT $PY_STR_PLOT $1 $2 &
wait

