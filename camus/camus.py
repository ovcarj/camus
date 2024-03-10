""" Definition of the Camus class.

This module is currently a placeholder for an interface
towards other components of the CAMUS library.

"""

import os
import importlib
import subprocess
import time
import glob
import numpy as np
import pickle
import shutil 

from ase import Atoms
from ase.io import write, read

from camus.batch import Batch
from camus.structures import Structures
from camus.stransition import STransition
from camus.stransitions import STransitions
from camus.tools.utils import save_to_pickle, load_pickle
from camus.tools.writers import Writers, write_lammps_data, write_POSCAR
from camus.tools.parsers import parse_lammps_dump, parse_sisyphus_xyz

scheduler_module = importlib.import_module('camus.scheduler')


class Camus:

    def __init__(self, structures=None, artn_parameters=None, lammps_parameters=None, sisyphus_parameters=None,
            scheduler='Slurm', dft_engine='VASP'):
        """
        Initializes a new Camus object, whose attributes `self.Cname` are instances of `name` classes.
        The methods in this class should allow for an interface towards the `name` classes.

        """
        
        self.Cstructures = Structures(structures)
        self.Cwriters = Writers(artn_parameters, lammps_parameters, 
                sisyphus_parameters, dft_engine)
        self.Cbatch = Batch(structures, artn_parameters, lammps_parameters, sisyphus_parameters,
                scheduler, dft_engine)

        scheduler_class = getattr(scheduler_module, scheduler)
        self.Cscheduler = scheduler_class()


    def initialize_STransitions(self, sisyphus_dictionary=None, base_directory=None, sisyphus_dictionary_path=None,
            sisyphus_analysis_directory=None, save_analysis=False, **kwargs):
        """
        Initializes an STransitions() instance to self.Cstransitions. 
        """

        self.Cstransitions = STransitions(sisyphus_dictionary=sisyphus_dictionary, base_directory=base_directory, sisyphus_dictionary_path=sisyphus_dictionary_path,
            sisyphus_analysis_directory=sisyphus_analysis_directory, save_analysis=False, **kwargs)

