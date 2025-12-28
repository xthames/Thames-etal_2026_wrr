#!/bin/bash
#SBATCH --mail-user=ayt5134@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=azh5924_cr_default
#SBATCH --partition=standard
#SBATCH --job-name=statecu_AnalysisBuild
#SBATCH --output=/storage/home/ayt5134/work/research/StateCU/jobs/output/%x.out
#SBATCH --error=/storage/home/ayt5134/work/research/StateCU/jobs/error/%x.err
#SBATCH --time=12:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40


# (0a) node list
NODE_LIST=($(scontrol show hostname $SLURM_JOB_NODELIST))


# (0b) load in the appropriate modules, venvs
module load anaconda
conda activate statecu_env
module load parallel


# (1) build experiment analysis per SOW and realization
PY_STR_SR="python /storage/home/ayt5134/work/research/StateCU/scripts/BuildAnalysisPerSR.py"
for (( NODE_IDX=0; NODE_IDX<${#NODE_LIST[@]}; NODE_IDX++ ))
do
	SOTWS_SEQ_SR=$(seq $NODE_IDX ${#NODE_LIST[@]} $(($1-1)))
	REALIZATIONS_SEQ_SR=$(seq 0 $(($2-1)))
	SRUN_OPTS_SR="--exclusive --nodes=1 --nodelist=${NODE_LIST[$NODE_IDX]} --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
	PARALLEL_OPTS_SR="--joblog /storage/home/ayt5134/work/research/StateCU/jobs/parallel/statecu_BuildSR_N$NODE_IDX.log --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"	
	srun $SRUN_OPTS_SR parallel $PARALLEL_OPTS_SR $PY_STR_SR ::: $SOTWS_SEQ_SR ::: $REALIZATIONS_SEQ_SR &
done
wait


# (2) build experiment analysis per user
PY_STR_USER="python /storage/home/ayt5134/work/research/StateCU/scripts/BuildAnalysisPerUser.py"
for (( NODE_IDX=0; NODE_IDX<${#NODE_LIST[@]}; NODE_IDX++ ))
do
	USER_SEQ=$(seq $NODE_IDX ${#NODE_LIST[@]} $(($3-1)))
	SRUN_OPTS_USER="--exclusive --nodes=1 --nodelist=${NODE_LIST[$NODE_IDX]} --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
	PARALLEL_OPTS_USER="--joblog /storage/home/ayt5134/work/research/StateCU/jobs/parallel/statecu_BuildUser_N$NODE_IDX.log --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"	
	srun $SRUN_OPTS_USER parallel $PARALLEL_OPTS_USER $PY_STR_USER ::: $1 ::: $2 ::: $USER_SEQ &
done
wait


