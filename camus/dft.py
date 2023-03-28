""" Definition of the DFT class.

This module defines everything regarding the DFT input.

"""

import os

from abc import ABC, abstractmethod
from ase.io import read

class DFT(ABC):

    def __init__(self, dft_parameters=None):
        """
        Initializes a new DFT object.

        Parameters:
            dft_parameters: INCAR parameters (currently limited purely to a handful of parameters we deemed necessary for a succesful yet fast SCF calculation)

        """

        if dft_parameters is not None:
            self._dft_parameters = dft_parameters
        else:
            self._dft_parameters = {}

    @abstractmethod
    def set_dft_parameters(self, **kwargs):
        ...

class VASP(DFT):

    def __init__(self, dft_parameters=None):

        super().__init__(dft_parameters)

    @property
    def dft_parameters(self):
        return self._dft_parameters

    @dft_parameters.setter
    def dft_parameters(self, new_dft_parameters):
        self._dft_parameters = {} # Probably makes more sense this way...
        self._dft_parameters = new_dft_parameters

    @dft_parameters.deleter
    def dft_parameters(self):
        del self._dft_parameters

    def set_dft_parameters(self, **kwargs):
        """ 
        Method that sets parameters to be written in the 
        INCAR file to self._dft_parameters dictionary.
        
        """
       
        default_parameters= {
            'SPACING': 0.25,
            'EDIFF': '1E-6',
            'ENCUT': 230,
            'ISMEAR': 0,
            'SIGMA': 0.05,
            'PREC': 'Accurate',
            'ALGO': 'Normal',
            'NELMIN': 6,
            'ISYM': 0,
            'LREAL': 'Auto',
            'LWAVE': '.FALSE.',
            'LCHARGE': '.FALSE.',
            'NCORE': 8,
            'METAGGA': 'R2SCAN',
            'LASPH': '.TRUE.',
            'LMIXTAU': '.TRUE.'
            }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self._dft_parameters
        for key, value in default_parameters.items():
            self._dft_parameters[key] = kwargs.pop(key, value)
    
    @staticmethod
    def write_POSCAR(input_structure, target_directory=None):
        """ Method which writes standart VASP input structure file -- POSCAR, using ASE Atoms object (input_structure MUST be provided).
        'target_directory' defaultly (if not specified otherwise) set to 'CAMUS_DFT_DIR'
        
        """
 

        # Set default target_directory 
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_DFT_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_DFT_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Write the POSCAR file to the target directory
        from ase.io import write
        write(os.path.join(target_directory, 'POSCAR'), input_structure, format='vasp')

    @staticmethod
    def write_VASP_sub(target_directory=None, job_filename='sub.sh'):
        """Temporary mathod which crates a sub.sh submission script specific to VASP calculation.
        
        """
        # Set default target_directory 
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_DFT_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_DFT_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Write the submission script to the target directory
        with open(os.path.join(target_directory, job_filename), 'w') as f:
            f.write(f"""!/bin/bash                                                          
#SBATCH --partition=cm
#SBATCH --job-name=vasp-matula
#SBATCH --cpus-per-task=1
#SBATCH --mem=200gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=1-20

module purge
module load VASP/6.3.1

export MKL_CBWR="AVX2"
export I_MPI_FABRICS=shm:ofi
ulimit -s unlimited


mpiexec.hydra -bootstrap slurm -n $SLURM_NTASKS vasp_std > log""")

