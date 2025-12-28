#!/bin/bash
#SBATCH --mail-user=ayt5134@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=azh5924_b
#SBATCH --partition=sla-prio
#SBATCH --job-name=statecu_NASACMIP6DataReader
#SBATCH --output=/storage/home/ayt5134/work/research/StateCU/jobs/output/%x.out
#SBATCH --error=/storage/home/ayt5134/work/research/StateCU/jobs/error/%x.err
#SBATCH --time=2:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=35


# load the modules, venvs
module load anaconda
conda activate statecu_env
module load parallel


# list of the pathways
PATHWAYS=("historical" "ssp126" "ssp245" "ssp370" "ssp585")
MODELNUM_SEQ=$(seq 35)
PY_STR="python /storage/home/ayt5134/work/research/StateCU/scripts/CMIP6NASADataReader.py"


## allocation permits 200 CPU max -- looping to distribute approx. that many CPUs across multiple nodes
NODE_LIST=($(scontrol show hostname $SLURM_JOB_NODELIST))
for (( NODE_IDX=0; NODE_IDX<${#NODE_LIST[@]}; NODE_IDX++ ))
do
	SRUN_OPTS="-exclusive --nodes=1 --nodelist=${NODE_LIST[$NODE_IDX]} --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
	JOBLOG_PATH="/storage/home/ayt5134/work/research/StateCU/jobs/parallel/${SLURM_JOB_NAME}_N$NODE_IDX.log"
	PARALLEL_OPTS="--joblog $JOBLOG_PATH --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"		

	# the command to run things in parallel
	srun $SRUN_OPTS parallel $PARALLEL_OPTS $PY_STR ::: ${PATHWAYS[$NODE_IDX]} ::: $MODELNUM_SEQ &
done
wait

