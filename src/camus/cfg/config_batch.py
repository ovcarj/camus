import sys
import click

import camus.utils.log as camus_log

from camus.cfg.config import Config

class ConfigBatch(Config):
    """
    Class which handles creation, deletion, reading and editing of the 
    batch config files.
    """

    def __init__(self, batch_config_path):
        """
        Check for the existence of the config file. If it exists, read its contents.

        Parameters
        ----------
        batch_config_path : str
            Path to the batch config file

        """

        batch_cfg_split = batch_config_path.rpartition('/')

        config_dir = batch_cfg_split[0]
        config_name = batch_cfg_split[-1]

        super().__init__(config_dir=config_dir, config_name=config_name,  
                help_message='Check "camus batch config edit --help" for instructions on editing the batch config file.')

    def define_default_values(self):
        """
        Defines the default values for a batch config file.

        """

        default_calculation_type = ''

        default_lammps_exe = ''
        default_lammps_run_command = ''
        default_lammps_flags = ''

        default_scheduler = ''

        self._config['CALCULATION'] = {
                'calc_type_placeholder': default_calculation_type
                }

        self._config['LAMMPS'] = {
                'lammps_exe': default_lammps_exe,
                'lammps_run_command': default_lammps_run_command,
                'lammps_flags': default_lammps_flags
                }

        self._config['Scheduler'] = {'scheduler': default_scheduler}

    def _config_wizard(self):
        """
        Procedure to guide the user after the initialization of the default config file.
        """

        batch_setup = camus_log.ask_yes_no(f'Do you wish to create a batch-wide configuration now? [Y/n]\n')

        if batch_setup == 'Y':

            lammps_setup = camus_log.ask_yes_no(f'Do you wish to create a batch-wide LAMMPS configuration now? [Y/n]\n')

            if lammps_setup == 'Y':
                self._lammps_wizard()
                click.echo(self._dashes)

            else:
                pass

            click.echo(f'This is a placeholder message to warn that currently, only the Slurm scheduler is implemented.')

            click.echo(self._dashes)
        
        else:
            click.echo('Using default batch configuration.')

        click.echo(f'Batch configuration successful!')
        click.echo(f'To edit the active batch config file, see camus batch config --help')
        click.echo(self._dashes)
