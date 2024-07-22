import abc

import os
import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

from camus.calc.calc_creator import CalcCreator

class LAMMPSCreator(CalcCreator, metaclass=abc.ABCMeta):
    """
    This class implements some methods of CalcCreator which are common to all LAMMPS calculations.

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

    def _define_required_engine_cfg(self):
        """
        Defines a dictionary of the engine-related config options that must be provided for the calculation to be created.

        """

        self._required_engine_cfg = {

                'LAMMPS_SETUP': ['lammps_exe']

                }

    def _define_env_cfg(self):
        """
        Defines dictionaries of the environment-related config options that can optionally be provided.

        """

        self._env_cfg = {

                'modules': {'LAMMPS_SETUP': 'lammps_modules'},
                'path': {'LAMMPS_SETUP': 'lammps_path_prepend'},
                'ld_path': {'LAMMPS_SETUP': 'lammps_ld_path_prepend'}

                }
