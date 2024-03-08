""" Definition of the DFT class.

This module defines everything regarding the DFT input.

"""

import os
import numpy as np

from abc import ABC, abstractmethod

from ase import Atom
from ase.io import read
from ase.io import write

from camus.structures import Structures

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
