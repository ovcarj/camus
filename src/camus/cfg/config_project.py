import sys
import click

import camus.utils.log as camus_log

from camus.cfg.config import Config

class ConfigProject(Config):
    """
    Class which handles creation, deletion, reading and editing of the 
    config file of the currently active project.
    """

    def __init__(self, project_config_path):
        """
        Check for the existence of the config file. If it exists, read its contents.

        Parameters
        ----------
        project_config_path : str
            Path to the project config file

        """

        project_cfg_split = project_config_path.rpartition('/')

        config_dir = project_cfg_split[0]
        config_name = project_cfg_split[-1]

        super().__init__(config_dir=config_dir, config_name=config_name,  
                help_message='Check "camus project config edit --help" for instructions on editing the project config file.')

    def define_default_values(self):
        """
        Defines the default values for a project config file.

        """

        default_lammps_exe = ''
        default_lammps_run_command = ''
        default_lammps_flags = ''
        default_scheduler = ''

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

        project_setup = camus_log.ask_yes_no(f'Do you wish to create a project-wide configuration now? [Y/n]\n')

        if project_setup == 'Y':

            lammps_setup = camus_log.ask_yes_no(f'Do you wish to create a project-wide LAMMPS configuration now? [Y/n]\n')

            if lammps_setup == 'Y':
                self._lammps_wizard()
                click.echo(self._dashes)

            else:
                pass

            click.echo(f'This is a placeholder message to warn that currently, only the Slurm scheduler is implemented.')

            click.echo(self._dashes)
        
        else:
            click.echo('Using default project configuration.')

        click.echo(f'Project configuration successful!')
        click.echo(f'To edit the active project config file, see camus project config --help')
        click.echo(self._dashes)
