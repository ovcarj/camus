import abc

import os
import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

from camus.calc.LAMMPS.energy_force.lammps_enfor_creator import LAMMPSEnForCreator

class LAMMPSClEnForCreator(LAMMPSEnForCreator):
    """
    LAMMPS single point energy-force calculation, using a classical potential.

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

    def _define_required_calc_cfg(self):
        """
        Defines a dictionary of the calculation-related config options that must be provided for the calculation to be created.

        """
        
        self._required_calc_cfg = {

                'CALCULATION': [
                    'calculation_type',
                    'calculation_method',
                    'calculation_engine',
                    'structures_file',
                    'lammps_input_file'
                    ]

                }

    def _check_required_cfg(self):
        pass
