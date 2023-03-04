#/bin/bash

# Bash script to automatically define all necessary variables to run CAMUS in .bashrc

# Define python path
WD=$(pwd)
echo 'export PYTHONPATH='$WD':$PYTHONPATH' >> ~/.bashrc

# Define path to the Camus directory
echo 'export CAMUS_BASE='$WD'' >> ~/.bashrc

# Define default LAMMPS data target directory
echo 'export CAMUS_LAMMPS_DATA_DIR=~/.CAMUS/LAMMPS_data' >> ~/.bashrc

# Define default ARTn data target directory
echo 'export CAMUS_ARTN_DATA_DIR=~/.CAMUS/ARTn_data' >> ~/.bashrc
