import sys
import os
import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

import platformdirs
import configparser

class Config:
    """
    Class which handles creation, deletion, reading and editing of the
    main ``camus`` config file, which is found at 
    ``platformdirs.user_config_dir(appname='camus')/camus.cfg``.
    """

    def __init__(self):
        """
        Check for the existence of the config file. If it exists, read its contents.
        """

        self._dashes = camus_log.get_log_dashes()

        self._config = configparser.ConfigParser()

        self._config_dir = platformdirs.user_config_dir(appname='camus')
        self._config_path = f'{self._config_dir}/camus.cfg'

        self.check_existence()

        if self._config_exists:
            self.read_config()

    def check_existence(self):
        """
        Checks the existence of the ``self.config`` directory; 
        the ``self._config_exists`` attribute is updated accordingly.
        """

        self._config_exists = camus_utils.file_exists(self._config_path)

    def define_default_values(self):
        """
        Defines the default values for the camus config file.
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

    def create_config_file(self):
        """
        Creates the ``camus`` config file. If ``self._config_path`` already exists, 
        the user is prompted if they want to proceed with the current config file.
        """

        self.check_existence()

        if self._config_exists:

            click.echo(f'WARNING: camus config file already exists at {self._config_path}')
            click.echo(self._dashes)
            self.handle_config_exists()

        else:

            self.define_default_values()

            if not camus_utils.directory_exists(self._config_dir):
                os.makedirs(self._config_dir)

            try:
                with open(self._config_path, 'w+') as configfile:
                    self._config.write(configfile)

            except:
                click.echo('Failed to initialize the camus config file at {self._config_path}. Exiting.')
                sys.exit()

            self.check_existence()

            if self._config_exists:

                click.echo(f'Path to the camus configuration file: {self._config_path}')
                click.echo(self._dashes)
                self.read_config()

            else:
                click.echo('Failed to initialize the camus config file at {self._config_path}. Exiting.')
                sys.exit()

            click.echo(f'camus will store all data in a given base directory.')
            click.echo(f'The default directory is: {self._base_directory}\n')

            new_path = input(f'Press "Enter" to keep the default or provide another path:\n')

            if len(new_path) > 0:
                self.edit_config_file(update_dict={'camusDataDirectory': {'data_directory': new_path}})

            click.echo(f'camus uses LAMMPS to run ML models.')

            new_lammps_exe = input(f'Enter the path to the LAMMPS executable or press "Enter" if you wish to provide the path later.\n')

            if len(new_lammps_exe) > 0:
                self.edit_config_file(update_dict={'LAMMPS': {'lammps_exe': new_lammps_exe}})

            new_lammps_run_command = input(f'Provide a command which will be used as a default to run the LAMMPS executable (e.g. mpirun -np 4) or press "Enter" to skip this step.\n')

            if len(new_lammps_run_command) > 0:
                self.edit_config_file(update_dict={'LAMMPS': {'lammps_run_command': new_lammps_run_command}})

            new_lammps_flags = input(f'Provide flags you wish to use to run LAMMPS (e.g. -sf omp -pk omp 1) or press "Enter" to not append any flags.\n')

            if len(new_lammps_flags) > 0:
                self.edit_config_file(update_dict={'LAMMPS': {'lammps_flags': new_lammps_flags}})

            click.echo(f'This is a placeholder message to warn that currently, only the Slurm scheduler is implemented...')

            click.echo(self._dashes)
            click.echo(f'camus config successful!')
            click.echo(self._dashes)

            self.print_config()

    def handle_config_exists(self):
        """
        Handles the case when the user tries to run ``camus init`` with a preexisting config file.
        """
        
        self.print_config()
        
        input_ok = False

        while(input_ok == False):

            proceed = input(f'Do you want to proceed with the current config file? [Y/n]\n')

            click.echo(self._dashes)

            try:
                assert (proceed == 'Y' or proceed == 'n')
                input_ok = True

            except AssertionError:
                click.echo('Please enter "Y" or "n".')
                click.echo(self._dashes)

        if proceed == 'Y':

            click.echo('Proceeding with the current camus config file.')
            click.echo(self._dashes)

        elif proceed == 'n':

            click.echo(f'Stopping, as requested.')
            click.echo(f'Check "camus config --help" for instructions on editing the config file or run "camus clean config" to delete the config file.')
            click.echo(self._dashes)

            sys.exit()

    def read_config(self):
        """
        Reads the contents of the config file and stores the values to ``self._config``.
        """

        self.check_existence()

        if self._config_exists:

            self._config.read(self._config_path)

            self._base_directory = self._config['camusDataDirectory']['data_directory']

        else:
            click.echo('The camus config file does not exist.')

    def print_config(self):
        """
        Prints the contents of the config file.
        """

        self.check_existence()

        if self._config_exists:

            with open(self._config_path, 'r') as f:

                click.echo(f'Contents of the camus config file at {self._config_path}:')
                click.echo(self._dashes)
                click.echo(f.read(), nl=False)

                click.echo(self._dashes)

        else:
            click.echo('The camus config file does not exist.')

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
                        click.echo(f'Invalid configuration subsection {subsection}')

                    else:

                        self._config[section][subsection] = value
                        
                        with open(self._config_path, 'w') as configfile:
                            self._config.write(configfile)

                        click.echo(f'Config: [{section}]: {subsection} updated to {value}')
                        self.read_config()


    def clean_config(self):
        """
        Deletes the ``self._config_dir`` configuration directory.
        """

        if not camus_utils.directory_exists(self._config_dir):
            click.echo('Nothing to delete.')

        else:
            camus_utils.delete_directory(self._config_dir)
