#!/bin/bash
#SBATCH --mail-user=ayt5134@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=azh5924_cr_default
#SBATCH --partition=standard
#SBATCH --job-name=statecu_AnalysisReadIn
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


# (1) read in data from StateCU output, create files
PY_STR_READ="python /storage/home/ayt5134/work/research/StateCU/scripts/StateCUDataReader.py"
for (( NODE_IDX=0; NODE_IDX<${#NODE_LIST[@]}; NODE_IDX++ ))
do
	SOTWS_SEQ_READ=$(seq $NODE_IDX ${#NODE_LIST[@]} $(($1-1)))
	REALIZATIONS_SEQ_READ=$(seq 0 $(($2-1)))
	SRUN_OPTS_READ="--exclusive --nodes=1 --nodelist=${NODE_LIST[$NODE_IDX]} --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
	PARALLEL_OPTS_READ="--joblog /storage/home/ayt5134/work/research/StateCU/jobs/parallel/statecu_ReadIn_N$NODE_IDX.log --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"	
	srun $SRUN_OPTS_READ parallel $PARALLEL_OPTS_READ $PY_STR_READ ::: $SOTWS_SEQ_READ ::: $REALIZATIONS_SEQ_READ &
done
wait

