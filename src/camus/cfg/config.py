import abc

import sys
import os
import click

import configparser

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

from camus.utils.environment import Environment

class Config(abc.ABC):
    """
    Base class which handles creation, deletion, reading and editing of configuration files.

    """

    def __init__(self, config_dir, config_name, help_message=''):
        """
        Check for the existence of the config file. If it exists, read its contents.

        Parameters
        ----------
        config_dir : str
            Directory in which the config file is or will be created
        config_name : str
            Name of the config file
        help_message : str
            Message that describes how to edit/delete the config file using the CLI 
        """

        self._dashes = camus_log.get_log_dashes()

        self._config = configparser.ConfigParser()

        if not config_dir.endswith('/'):
            config_dir += '/'

        self._config_dir = config_dir
        self._config_path = f'{self._config_dir}{config_name}'

        self.check_existence()

        if self._config_exists:
            self.read_config()

        self._help_message = help_message

        self._env = Environment()

    def check_existence(self):
        """
        Checks the existence of the ``self.config`` directory; 
        the ``self._config_exists`` attribute is updated accordingly.

        """

        self._config_exists = camus_utils.file_exists(self._config_path)

    @abc.abstractmethod
    def define_default_values(self):
        """
        Defines the default values for the config file.

        The values should be defined in the following way:

        self._config['SECTION'] = {'option0': value0, option1: value1, ...}

        """

        pass

    def create_config_file(self):
        """
        Creates the config file with default values. If ``self._config_path`` already exists, 
        the user is prompted if they want to proceed with the current config file.

        """

        self.check_existence()

        if self._config_exists:

            click.echo(f'WARNING: config file already exists at {self._config_path}')
            click.echo(self._dashes)
            self._handle_config_exists()

        else:

            self.define_default_values()

            if not camus_utils.directory_exists(self._config_dir):
                os.makedirs(self._config_dir)

            try:
                with open(self._config_path, 'w+') as configfile:
                    self._config.write(configfile)

            except:
                click.echo('Failed to initialize the config file at {self._config_path}. Do you have the required permissions to write to the requested directory? Exiting.')
                sys.exit()

            self.check_existence()

            if self._config_exists:

                click.echo(f'Path to the configuration file: {self._config_path}')
                click.echo(self._dashes)
                self.read_config()

            else:
                click.echo('Failed to initialize the config file at {self._config_path}. Exiting.')
                sys.exit()

            self._config_wizard()

            self.print_config()
            
    def _handle_config_exists(self):
        """
        Handles the case when the user tries to run ``self.create_config_file()`` with a preexisting config file.

        """
        
        self.print_config()
        
        input_ok = False

        proceed = camus_log.ask_yes_no('Do you want to proceed with the current config file? [Y/n]\n')

        if proceed == 'Y':

            click.echo('Proceeding with the current config file.')
            click.echo(self._dashes)

        elif proceed == 'n':

            click.echo(f'Stopping, as requested.')
            click.echo(self._help_message)
            click.echo(self._dashes)

            sys.exit()

    def read_config(self):
        """
        Reads the contents of the config file and stores the values to ``self._config``.
        """

        self.check_existence()

        if self._config_exists:
            self._config.read(self._config_path)

        else:
            click.echo('The config file does not exist.')

    def print_config(self):
        """
        Prints the contents of the config file.

        """

        self.check_existence()

        if self._config_exists:

            with open(self._config_path, 'r') as f:

                click.echo(f'Contents of the config file at\n{self._config_path}')
                click.echo(self._dashes)
                click.echo(f.read(), nl=False)

                click.echo(self._dashes)

        else:
            click.echo('The config file does not exist.')

    def edit_config_file(self, update_dict):
        """
        Edit the contents of the config file.

        The update_dict should be of form {section0: {subsection00: value00, subsection01: value01}, section1: {subsection10: value10, subsection11: value11}, ...},
        where the sections and subsections correspond to the configuration file.

        Parameters
        ----------
        update_dict : dict
            Dictionary of form {section0: {subsection00: value00, subsection01: value01}, section1: {subsection10: value10, subsection12: value12}, ...}

        """

        self.read_config()

        all_sections = self._config.sections()

        for section, subsections_values in update_dict.items():

            if section not in all_sections:
                click.echo(f'Invalid configuration section {section}')

            else:

                all_subsections = self._config[section].keys()

                for subsection, value in subsections_values.items():

                    if subsection not in all_subsections:
                        click.echo(f'Invalid configuration option {subsection}')

                    else:

                        self._config[section][subsection] = value
                        
                        with open(self._config_path, 'w') as configfile:
                            self._config.write(configfile)

                        click.echo(f'Edited {self._config_path}\n')
                        click.echo(f'[{section}]: {subsection} updated to {value}')
                        self.read_config()

    def edit_config_by_subsection(self, subsection, value):
        """
        Edits the contents of the config file by passing only the subsection and the value, while the section is automatically found.

        Parameters
        ----------
        subsection : str
            Subsection in the config file
        value : str | float | int
            Value that the subsection is updated to

        """

        self.read_config()
        all_sections = self._config.sections()

        update_dict = {}

        for section in all_sections:

            if subsection in self._config.options(section):

                update_dict[section] = {subsection: value}
                self.edit_config_file(update_dict=update_dict)

                break
            
        else:
            click.echo(f'Invalid config option "{subsection}"')

    @abc.abstractmethod
    def _config_wizard(self):
        """
        A procedure to guide the user through editing the config file after it was initialized.
        """
        pass

    def _lammps_setup_wizard(self):
        """
        A procedure to guide the user through LAMMPS setup.

        """

        click.echo('Starting LAMMPS setup...')
        click.echo(self._dashes)

        new_lammps_exe = input(f'Enter the path to the LAMMPS executable or press "Enter" if you wish to provide the path later.\n')

        if len(new_lammps_exe) > 0:
            self.edit_config_file(update_dict={'LAMMPS_SETUP': {'lammps_exe': new_lammps_exe}})
            click.echo('\n')

        if self._env._has_modules:

            new_lammps_modules = input(f'Enter a comma-separated list of modules you wish to load when running LAMMPS or press "Enter" to skip this step.\n')

            if len(new_lammps_modules) > 0:
                self.edit_config_file(update_dict={'LAMMPS_SETUP': {'lammps_modules': new_lammps_modules}})
                click.echo('\n')
        else:
            click.echo('No ``module`` program was found on the OS. No additional modules will be loaded while running LAMMPS.\n')

        new_lammps_path_prepend = input(f'Enter a colon-separated list of paths which will be prepended to the PATH environment variable when running LAMMPS or press "Enter" to skip this step.\n')

        if len(new_lammps_path_prepend) > 0:
            self.edit_config_file(update_dict={'LAMMPS_SETUP': {'lammps_path_prepend': new_lammps_path_prepend}})
            click.echo('\n')

        if self._env._is_unix:

            new_lammps_ld_path_prepend = input(f'Enter a colon-separated list of paths which will be prepended to the LD_LIBRARY_PATH environment variable when running LAMMPS or press "Enter" to skip this step.\n')

            if len(new_lammps_ld_path_prepend) > 0:
                self.edit_config_file(update_dict={'LAMMPS_SETUP': {'lammps_ld_path_prepend': new_lammps_ld_path_prepend}})
                click.echo('\n')

        click.echo(self._dashes)
        click.echo('LAMMPS setup finished!')

    def _scheduler_wizard(self):
        """
        A procedure to guide the user through the scheduler setup.

        """

        click.echo('Starting scheduler setup...')
        click.echo(self._dashes)

        click.echo(f'This is a placeholder message to warn that currently, only the Slurm scheduler is implemented.\n')

        new_partition = input(f'Enter the name of the default cluster partition to be used or press "Enter" to skip this step.\n')

        if len(new_partition) > 0:
            self.edit_config_file(update_dict={'SCHEDULER': {'partition': new_partition}})
            click.echo('\n')

        new_memory = input(f'Enter the default amount of memory a job will request in the format that the scheduler can read or press "Enter" to skip this step.\n')

        if len(new_memory) > 0:
            self.edit_config_file(update_dict={'SCHEDULER': {'memory': new_memory}})
            click.echo('\n')

        new_nodes = input(f'Enter the default number of nodes a job will request in the format that the scheduler can read or press "Enter" to skip this step.\n')

        if len(new_nodes) > 0:
            self.edit_config_file(update_dict={'SCHEDULER': {'nodes': new_nodes}})
            click.echo('\n')

        new_walltime = input(f'Enter the value of the default job walltime in the format that the scheduler can read or press "Enter" to skip this step.\n')

        if len(new_walltime) > 0:
            self.edit_config_file(update_dict={'SCHEDULER': {'walltime': new_walltime}})
            click.echo('\n')

        new_additional_commands = input(f'Enter a comma-separated list of commands to be written into a submission script (e.g., export MKL_CBWR="AVX2", export I_MPI_FABRICS=shm:ofi) or press "Enter" to skip this step.\n')

        if len(new_additional_commands) > 0:
            self.edit_config_file(update_dict={'SCHEDULER': {'additional_scheduler_commands': new_additional_commands}})
            click.echo('\n')

        click.echo(self._dashes)
        click.echo('Scheduler setup finished!')

    def _mpi_wizard(self):
        """
        A procedure to guide the user through the MPI setup.

        """

        click.echo('Starting MPI setup...')
        click.echo(self._dashes)

        new_mpi_command = input(f'Enter default command for running MPI programs (e.g. mpirun) or press "Enter" to skip this step.\n')

        if len(new_mpi_command) > 0:
            self.edit_config_file(update_dict={'MPI': {'mpi_command': new_mpi_command}})
            click.echo('\n')

        new_mpi_flags = input(f'Provide default flags you wish to append when running an MPI application (e.g. -sf omp -pk omp 1) or press "Enter" not to append any flags.\n')

        if len(new_mpi_flags) > 0:
            self.edit_config_file(update_dict={'MPI': {'mpi_flags': new_mpi_flags}})
            click.echo('\n')

        click.echo(self._dashes)
        click.echo('MPI setup finished!')

    def clean_config(self):
        """
        Deletes the file at ``self._config_path``.

        """

        camus_utils.delete_file(self._config_path)
