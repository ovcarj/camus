#/bin/bash

# Bash script to automatically define all necessary variables to run CAMUS in .bashrc
# LAMMPS_EXE variable must be set
# RUN_LAMMPS variable (e.g. mpirun -np 8) should be made optional (not yet tested)

echo '' >> ~/.bashrc
echo '# >>> Initialize CAMUS environment variables >>>' >> ~/.bashrc
echo '' >> ~/.bashrc

# Define python path
WD=$(pwd)
echo 'export PYTHONPATH='$WD':$PYTHONPATH' >> ~/.bashrc

# Define path to the Camus directory
echo 'export CAMUS_BASE='$WD'' >> ~/.bashrc

# Define default LAMMPS data target directory
echo 'export CAMUS_LAMMPS_DATA_DIR=~/.CAMUS/LAMMPS_data' >> ~/.bashrc

# Define default ARTn data target directory
echo 'export CAMUS_SISYPHUS_DATA_DIR=~/.CAMUS/SISYPHUS_data' >> ~/.bashrc

# Define LAMMPS execution command
echo 'export RUN_LAMMPS=""' >> ~/.bashrc

# Define path to LAMMPS executable 
echo 'export LAMMPS_EXE=/path/to/lammps/executable' >> ~/.bashrc

echo '' >> ~/.bashrc
echo '# <<< Initialize CAMUS environment variables <<<' >> ~/.bashrc
