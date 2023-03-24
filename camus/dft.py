""" Definition of the DFT class.

This module does fuck all right now.

- create dft input structure
-- specifically VASP input -- POSCAR
-[ab] set dft calculation parameters
-[ab] assemble DFT calculation (POSCAR - from create_dft_input, INCAR - based on the parameters, POTCAR - defaultly what Ivor provided (introduce env variable for default) )
- (run)
-[ab] parse dft output (OUTCAR -> .traj)

-- add env variable for deffault DFT directory for testing purposes

!! we should add tracking of the DFT process and check for convergence, I have a bash script for this !!
"""

import os

from abc import ABC, abstractmethod
from ase.io import read

class DFT(ABC):

    def __init__(self, dft_parameters=None):
        """
        Initializes a new DFT object.

        Parameters:
            dft_parameters: (...)

        """

        if dft_parameters is not None:
            self._dft_parameters = dft_parameters
        else:
            self._dft_parameters = {}

        # Default directory for the whole DFT calculation (?)
        '''
                (...)
        '''

# make abstract methods the same as they are in VASP
    @abstractmethod
    def set_dft_parameters(self, **kwargs):
        ...

    @abstractmethod
    def assemble_dft_calculation()
        ...

    @abstractmethod
    def parse_dft_output(self, target_directory=None)
        ...

class VASP(DFT):

    def __init__(self, dft_parameters=None):

        super().__init__(dft_parameters)

    @property
    def dft_parameters
        return self._dft_parameters

    @dft_parameters.setter
    def dft_parameters(self, new_dft_parameters):
        self._dft_parameters = {} # Probably makes more sense this way...
        self._dft_parameters = new_dft_parameters

    @dft_parameters.deleter
    def dft_parameters(self):
        del self._dft_parameters

    def set_dft_parameters(self, **kwargs):
        """ Method that sets parameters to be written in the 
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
    def write_POSCAR(input_structure, specorder=None):
'''
If you're calling this function you obviously want to make a POSCAR out of your trajectory so it would stand to reason you have an input structure.

Fucking specorder: from what I can tell the write_vasp method keeps the order from the input trajecotry file unless you specify differently (then it will chose alphabetical order), but seeing as the files throughout the camus algorithm are repeatedly ordered in out desired 'Br I Cs Pb', zet another ordering doesn't seem necessary. 
''' 
        data = read(self.input_structure)
        write(f'POSCAR', data, format='vasp')

        # target directory (?)

    def assemble_dft_calculation(self, target_directory=None):
    '''
    - POSCAR -- from write_POSCAR
    - POTCAR -- provide path or use the default path
    - write INCAR -- using the parameters (if not provided use the default) to create an INCAR file
    - create a directory including these three files [and then submit the job using the scheduler]
    '''
        
        # Set default directory 
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_DFT_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_DFT_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.dft_parameters:
            self.set_dft_parameters()

        # Define the INCAR file content
        INCAR_content = "&DFT_PARAMETERS\n"
        for key, value in self.dft_parameters.items():
            if value is not None:
                INCAR_content += f"  {key} = {self._dft_parameters[key]}\n"
        INCAR_content += "/\n"

        # Write the INCAR file to the target directory
        with open(os.path.join(target_directory, 'INCAR'), 'w') as f:
            f.write(artn_in_content)

        # The path to POTCAR


    # might also need directory paths
    def parse_dft_output(self, dft_output_file='OUTCAR', trajectory_name='dft_out.traj'):
        from ase.io.trajectory import Trajectory
        traj = Trajectory(self.trajectory_name, 'w')
        atoms = read(self.dft_output_file)
        traj.write(atoms)


         
