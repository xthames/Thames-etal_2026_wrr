#!/bin/bash


# get SOW, realization numbers
SCN_NUM=$(($2+1))
SIM_NUM=$(($3+1))

# filepaths
SYNTH_FILEPATH="/storage/work/ayt5134/research/StateCU/synthetic/$1/Scenario$SCN_NUM"
STATECU_FILEPATH="/storage/work/ayt5134/research/StateCU/cdss-dev"

# move the simulation .prc, .tem, .fd files into the corresponding folder
mv $SYNTH_FILEPATH/*Sim$SIM_NUM.* $SYNTH_FILEPATH/Sim$SIM_NUM

# change the filenames for the .prc, .tem, .fd files
mv $SYNTH_FILEPATH/Sim$SIM_NUM/*.prc $SYNTH_FILEPATH/Sim$SIM_NUM/simulation.prc
mv $SYNTH_FILEPATH/Sim$SIM_NUM/*.tem $SYNTH_FILEPATH/Sim$SIM_NUM/simulation.tem
mv $SYNTH_FILEPATH/Sim$SIM_NUM/*.fd $SYNTH_FILEPATH/Sim$SIM_NUM/simulation.fd

# copy .rcu file to SYNTH_FILEPATH
cp $STATECU_FILEPATH/cm2015_StateCU/TemplateCU/simulation.rcu $SYNTH_FILEPATH/Sim$SIM_NUM/

# create a symlink to StateCU.exe
ln -s $STATECU_FILEPATH/cdss-app-statecu-fortran/src/main/fortran/statecu-14.0.1-gfortran-lin-64bit $SYNTH_FILEPATH/Sim$SIM_NUM/statecu_exe

# change directory to this simulation file, run StateCU.exe with the synthetic scenario, and change back to the "home" directory
cd $SYNTH_FILEPATH/Sim$SIM_NUM && ./statecu_exe simulation && cd /storage/work/ayt5134/research/StateCU

