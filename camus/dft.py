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
import subprocess
import time

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

        # Set self._scheduler_parameters
        for key, value in default_parameters.items():
            self._scheduler_parameters[key] = kwargs.pop(key, value)
    
    @staticmethod
    def write_POSCAR(input_structure=None, specorder=None):
        #if input_structure None...

        from ase.io.vasp import write_vasp

        data = read(self.input_structure)
        poscar = write_vasp(f'POSCAR', data, vasp5=True)

    def assemble_dft_calculation(self):
    '''
    - POSCAR -- from write_POSCAR
    - POTCAR -- provide path or use the default path
    - write INCAR -- using the parameters (if not provided use the default) to create an INCAR file
    - create a directory including these three files [and then submit the job using the scheduler]
    '''

    def parse_dft_output(self, dft_output_file='OUTCAR', trajectory_name='dft_out.traj'):
        from ase.io.trajectory import Trajectory
        traj = Trajectory(self.trajectory_name, 'w')
        atoms = read(self.dft_output_file)
        traj.write(atoms)


         
