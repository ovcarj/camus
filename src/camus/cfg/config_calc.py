import sys
import click

import camus.utils.log as camus_log

from camus.cfg.config import Config

class ConfigCalc(Config):
    """
    Class which handles creation, deletion, reading and editing of
    calculation config files.

    """

    def __init__(self, calc_config_path):
        """
        Check for the existence of the calculation config file. If it exists, read its contents.

        Parameters
        ----------
        calc_config_path : str
            Path to the calculation config file

        """

        calc_cfg_split = batch_config_path.rpartition('/')

        config_dir = calc_cfg_split[0]
        config_name = calc_cfg_split[-1]

        super().__init__(config_dir=config_dir, config_name=config_name,  
                help_message='')

    def define_default_values(self):
        """
        Defines the default values for a calculation config file.

        """

        default_calc_dir = ''
        default_calc_label = ''
        default_energy_force_engine = ''
        default_calculation_type = ''

        default_path_to_structures = ''

        default_lammps_exe = ''
        default_lammps_run_command = ''
        default_lammps_flags = ''

        default_write_sub = 'yes'

        self._config['CALCULATION'] = {
                'calc_dir': default_calc_dir,
                'calc_label': default_calc_label,
                'energy_force_engine': default_energy_force_engine,
                'calculation_type': default_calculation_type
                }

        self._config['STRUCTURES'] = {
                'structures_file': default_path_to_structures
                }

        self._config['SCHEDULER'] = {
                'write_sub': default_write_sub
                }

        self._config['LAMMPS'] = {
                'lammps_exe': default_lammps_exe,
                'lammps_run_command': default_lammps_run_command,
                'lammps_flags': default_lammps_flags
                }

    def _config_wizard(self):
        """
        Procedure to guide the user after the initialization of the default config file.
        """
        pass
