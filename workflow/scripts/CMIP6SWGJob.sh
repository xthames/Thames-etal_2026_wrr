#!/bin/bash
#SBATCH --mail-user=ayt5134@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=azh5924_cr_default
#SBATCH --partition=standard
#SBATCH --job-name=statecu_CMIP6SWGJob
#SBATCH --output=/storage/home/ayt5134/work/research/StateCU/jobs/output/%x.out
#SBATCH --error=/storage/home/ayt5134/work/research/StateCU/jobs/error/%x.err
#SBATCH --time=3:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40


# (0a) node list
NODE_LIST=($(scontrol show hostname $SLURM_JOB_NODELIST))


# (0b) load in the appropriate modules, venvs
module load anaconda
conda activate statecu_env
module load parallel


# (1) extract all of the GMMHMM/copulae parameters from the data fitting step
PY_STR_EXTRACT="python /storage/home/ayt5134/work/research/StateCU/scripts/ExtractParameters.py"
SRUN_OPTS_EXTRACT="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=1"
srun $SRUN_OPTS_EXTRACT $PY_STR_EXTRACT $1 $2 &
wait


# (2) generate SOWs from extracted parameters
PY_STR_GEN="python /storage/home/ayt5134/work/research/StateCU/scripts/GenerateSOTWs.py"
for (( NODE_IDX=0; NODE_IDX<${#NODE_LIST[@]}; NODE_IDX++ ))
do
	SOTWS_SEQ_GEN=$(seq $NODE_IDX ${#NODE_LIST[@]} $(($2-1)))
	SRUN_OPTS_GEN="--exclusive --nodes=1 --nodelist=${NODE_LIST[$NODE_IDX]} --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
	PARALLEL_OPTS_GEN="--joblog /storage/home/ayt5134/work/research/StateCU/jobs/parallel/statecu_GenSOTWs_N$NODE_IDX.log --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"	
	srun $SRUN_OPTS_GEN parallel $PARALLEL_OPTS_GEN $PY_STR_GEN ::: $1 ::: $SOTWS_SEQ_GEN &
done
wait


# (3) synthesize precip/temp from the scaled NOAA observations and the scenario parameter fits
PY_STR_SYNTH="python /storage/home/ayt5134/work/research/StateCU/scripts/SynthesizePT.py"
for (( NODE_IDX=0; NODE_IDX<${#NODE_LIST[@]}; NODE_IDX++ ))
do
	SOTWS_SEQ_SYNTH=$(seq $NODE_IDX ${#NODE_LIST[@]} $(($2-1)))
	REALIZATIONS_SEQ_SYNTH=$(seq 0 $(($3-1)))
	SRUN_OPTS_SYNTH="--exclusive --nodes=1 --nodelist=${NODE_LIST[$NODE_IDX]} --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
	PARALLEL_OPTS_SYNTH="--joblog /storage/home/ayt5134/work/research/StateCU/jobs/parallel/statecu_Synthesize_N$NODE_IDX.log --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"	
	srun $SRUN_OPTS_SYNTH parallel $PARALLEL_OPTS_SYNTH $PY_STR_SYNTH ::: $1 ::: $SOTWS_SEQ_SYNTH ::: $REALIZATIONS_SEQ_SYNTH &
done
wait


# (4) move the synthetics to their corresponding folders and execute StateCU.exe
SH_STR_MCE="bash /storage/home/ayt5134/work/research/StateCU/scripts/StateCUexe.sh"
for (( NODE_IDX=0; NODE_IDX<${#NODE_LIST[@]}; NODE_IDX++ ))
do
	SOTWS_SEQ_MCE=$(seq $NODE_IDX ${#NODE_LIST[@]} $(($2-1)))
	REALIZATIONS_SEQ_MCE=$(seq 0 $(($3-1)))
	SRUN_OPTS_MCE="--exclusive --nodes=1 --nodelist=${NODE_LIST[$NODE_IDX]} --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
	PARALLEL_OPTS_MCE="--joblog /storage/home/ayt5134/work/research/StateCU/jobs/parallel/statecu_MCE_N$NODE_IDX.log --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"	
	srun $SRUN_OPTS_MCE parallel $PARALLEL_OPTS_MCE $SH_STR_MCE ::: $1 ::: $SOTWS_SEQ_MCE ::: $REALIZATIONS_SEQ_MCE &
done
wait

