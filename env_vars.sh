#/bin/bash

# Bash script to automatically define all necessary variables to run CAMUS in .bashrc

# Define python path
cd ..
WD=$(pwd)
echo 'export PYTHONPATH='$WD':$PYTHONPATH' >> ~/.bashrc
cd -
