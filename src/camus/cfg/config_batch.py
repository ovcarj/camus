import sys
import click

import camus.utils.log as camus_log

from camus.cfg.config import Config
from camus.cfg.config_project import ConfigProject

from camus.db.project import Project

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
        Defines the default values for a batch config file. The majority of the defaults are taken from the active project config file.

        """

        default_energy_force_engine = ''
        default_calculation_type = ''

        default_path_to_structures = ''

        self._config['CALCULATION'] = {
                'energy_force_engine': default_energy_force_engine,
                'calculation_type': default_calculation_type
                }

        self._config['STRUCTURES'] = {
                'structures_file': default_path_to_structures
                }

        proj = Project(load_active=True)
        proj_cfg = ConfigProject(proj.config)

        self._config['LAMMPS_SETUP'] = proj_cfg._config['LAMMPS_SETUP']
        self._config['MPI'] = proj_cfg._config['MPI']
        self._config['SCHEDULER'] = proj_cfg._config['SCHEDULER']

    def _config_wizard(self):
        """
        Procedure to guide the user after the initialization of the default config file.
        """

        batch_setup = camus_log.ask_yes_no(f'Do you wish to create a batch-wide configuration now? [Y/n]\n')

        if batch_setup == 'Y':

            lammps_setup = camus_log.ask_yes_no(f'Do you wish to create a batch-wide LAMMPS configuration now? [Y/n]\n')

            if lammps_setup == 'Y':
                self._lammps_setup_wizard()
                click.echo(self._dashes)

            else:
                pass

            click.echo(f'This is a placeholder message to warn that currently, only the Slurm scheduler is implemented.')

            click.echo(self._dashes)
        
        else:
            click.echo('Using the configuration from the active project.')

        click.echo(f'Batch configuration successful!')
        click.echo(f'To edit the active batch config file, see camus batch config --help')
        click.echo(self._dashes)
