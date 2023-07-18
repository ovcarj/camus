#/bin/bash

# Bash script to automatically define all necessary variables to run CAMUS in .bashrc
# LAMMPS_EXE variable must be set
# RUN_LAMMPS variable (e.g. mpirun -np 8) should be made optional (not yet tested)

echo '' >> ~/.bashrc
echo '# >>> Initialize CAMUS environment variables >>>' >> ~/.bashrc
echo '' >> ~/.bashrc

# Define python path
WD=$(pwd)
echo 'export PYTHONPATH='$WD':/optional_path/to/IRA_interface/:$PYTHONPATH' >> ~/.bashrc

# Define path to the Camus directory
echo 'export CAMUS_BASE='$WD'' >> ~/.bashrc

# Define default LAMMPS data target directory
echo 'export CAMUS_LAMMPS_DATA_DIR=~/.CAMUS/LAMMPS_data' >> ~/.bashrc

# Define default Sisyphus data target directory
echo 'export CAMUS_SISYPHUS_DATA_DIR=~/.CAMUS/SISYPHUS_data' >> ~/.bashrc

# Define default LAMMPS minimization target directory
echo 'export CAMUS_LAMMPS_MINIMIZATION_DIR=~/.CAMUS/LAMMPS_minimization' >> ~/.bashrc

# Define default DFT target directory
echo 'export CAMUS_DFT_DIR=~/.CAMUS/DFT' >> ~/.bashrc

# Define LAMMPS execution command
echo 'export RUN_LAMMPS=""' >> ~/.bashrc

# Define path to LAMMPS executable
echo 'export LAMMPS_EXE=/path/to/lammps/executable' >> ~/.bashrc

# Define additional LAMMPS flags to be used when running LAMMPS
echo 'export LAMMPS_FLAGS=""' >> ~/.bashrc

# Define default path to ML model
echo 'export ML_MODEL=/path/to/ml/model' >> ~/.bashrc

# Define default path to POTCAR 
echo 'export DFT_POTCAR=/path/to/dft/POTCAR' >> ~/.bashrc

echo '' >> ~/.bashrc
echo '# <<< Initialize CAMUS environment variables <<<' >> ~/.bashrc
