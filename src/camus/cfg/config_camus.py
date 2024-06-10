import sys
import click

import platformdirs

import camus.utils.log as camus_log

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
        Defines the default values for the main ``camus`` config file.

        """

        default_base = platformdirs.user_data_dir(appname='camus')
        
        default_active_project = ''

        default_mpi_command = ''
        default_mpi_flags = ''

        default_lammps_exe = ''
        default_lammps_modules = ''
        default_lammps_path_prepend = ''
        default_lammps_ld_path_prepend = ''

        default_scheduler = 'Slurm'
        default_partition = ''
        default_memory = ''
        default_nodes = ''
        default_walltime = ''
        default_scheduler_commands = ''

        self._config['camusDataDirectory'] = {'data_directory': default_base}

        self._config['ActiveProject'] = {'active_project': default_active_project}

        self._config['LAMMPS_SETUP'] = {
                'lammps_exe': default_lammps_exe,
                'lammps_modules': default_lammps_modules,
                'lammps_path_prepend': default_lammps_path_prepend,
                'lammps_ld_path_prepend': default_lammps_ld_path_prepend
                }

        self._config['MPI'] = {
                'mpi_command': default_mpi_command,
                'mpi_flags': default_mpi_flags
                }

        self._config['SCHEDULER'] = {
                'scheduler': default_scheduler,
                'partition': default_partition,
                'memory': default_memory,
                'nodes': default_nodes,
                'walltime': default_walltime,
                'additional_scheduler_commands': default_scheduler_commands
                }

    def _config_wizard(self):
        """
        Procedure to guide the user after the initialization of the default config file.
        """

        click.echo(f'camus will store all data in a given base directory.')
        click.echo(f'The default directory is: {self._config["camusDataDirectory"]["data_directory"]}\n')

        new_path = input(f'Press "Enter" to keep the default or provide another path:\n')

        if len(new_path) > 0:
            self.edit_config_file(update_dict={'camusDataDirectory': {'data_directory': new_path}})

        lammps_setup = camus_log.ask_yes_no(f'Do you wish to create a global LAMMPS configuration now? [Y/n]\n')

        if lammps_setup == 'Y':
            self._lammps_setup_wizard()
            click.echo(self._dashes)

        else:
            pass

        self._mpi_wizard()
        click.echo(self._dashes)

        self._scheduler_wizard()
        click.echo(self._dashes)

        click.echo(f'camus configuration successful!')
        click.echo(f'To edit the config file, see camus config edit --help')
        click.echo(self._dashes)

        click.echo(f'See `camus project --help` to create a new project or switch to an existing one.')
        click.echo(self._dashes)

