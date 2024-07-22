import abc

import os
import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

from camus.calc.LAMMPS.lammps_creator import LAMMPSCreator

class LAMMPSEnForCreator(LAMMPSCreator, metaclass=abc.ABCMeta):
    """
    Parent class for a LAMMPS energy-force calculation, since the method can be
    ``classical_potential`` or ``neural_network``, but inputs and sub scripts are the same.

    """

    def __init__(self, calc_config_path=None):
        """
        Get required config options. If the calculation config file exists, read its contents.

        Parameters
        ----------
        calc_config_path : None | str
            Path to the calculation config file

        """

        super().__init__(calc_config_path=calc_config_path)

    def write_inputs(self):
        """
        Writes all necessary input files for a calculation.

        """
        pass

    def write_submission_script(self):
        """
        Writes the appropriate submission script.

        """
        pass
