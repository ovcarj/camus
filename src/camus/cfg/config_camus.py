import sys
import click

import platformdirs

from camus.cfg.config import Config

class ConfigCamus(Config):
    """
    Class which handles creation, deletion, reading and editing of the
    main ``camus`` config file, which is found at 
    ``platformdirs.user_config_dir(appname='camus')/camus.cfg``.
    """

    def __init__(self):
        """
        Check for the existence of the config file. If it exists, read its contents.
        """

        config_dir = platformdirs.user_config_dir(appname='camus')

        super().__init__(config_dir=config_dir, config_name='camus.cfg',  
                help_message='Check "camus config edit --help" for instructions on editing the config file or run "camus clean config" to delete the current config file.')

    def define_default_values(self):
        """
        Defines the default values for the main camus config file.
        """

        default_base = platformdirs.user_data_dir(appname='camus')
        default_lammps_exe = '/path/to/lammps/executable'
        default_lammps_run_command = ''
        default_lammps_flags = ''
        default_scheduler = 'Slurm'

        self._config['camusDataDirectory'] = {'data_directory': default_base}

        self._config['LAMMPS'] = {
                'lammps_exe': default_lammps_exe,
                'lammps_run_command': default_lammps_run_command,
                'lammps_flags': default_lammps_flags
                }

        self._config['Scheduler'] = {'scheduler': default_scheduler}

        self._config['ActiveProject'] = {'active_project': ''}

    def _config_wizard(self):
        """
        Procedure to guide the user after the initialization of the default config file.
        """

        click.echo(f'camus will store all data in a given base directory.')
        click.echo(f'The default directory is: {self._config["camusDataDirectory"]["data_directory"]}\n')

        new_path = input(f'Press "Enter" to keep the default or provide another path:\n')

        if len(new_path) > 0:
            self.edit_config_file(update_dict={'camusDataDirectory': {'data_directory': new_path}})

        click.echo(f'camus uses LAMMPS to run ML models.')

        new_lammps_exe = input(f'Enter the path to the LAMMPS executable or press "Enter" if you wish to provide the path later.\n')

        if len(new_lammps_exe) > 0:
            self.edit_config_file(update_dict={'LAMMPS': {'lammps_exe': new_lammps_exe}})

        click.echo('\n')

        new_lammps_run_command = input(f'Provide a command which will be used as a default to run the LAMMPS executable (e.g. mpirun -np 4) or press "Enter" to skip this step.\n')

        if len(new_lammps_run_command) > 0:
            self.edit_config_file(update_dict={'LAMMPS': {'lammps_run_command': new_lammps_run_command}})

        click.echo('\n')

        new_lammps_flags = input(f'Provide flags you wish to use to run LAMMPS (e.g. -sf omp -pk omp 1) or press "Enter" to not append any flags.\n')

        if len(new_lammps_flags) > 0:
            self.edit_config_file(update_dict={'LAMMPS': {'lammps_flags': new_lammps_flags}})

        click.echo('\n')

        click.echo(f'This is a placeholder message to warn that currently, only the Slurm scheduler is implemented.')

        click.echo('\n')

        click.echo(f"""Currently, no project is active. See `camus project --help` to create a new project or switch to an existing one.""")

        click.echo(self._dashes)
        click.echo(f'camus configuration successful!')
        click.echo(f'To edit the config file, see camus config edit --help')
        click.echo(self._dashes)
