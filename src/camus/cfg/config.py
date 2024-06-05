import abc

import sys
import os
import click

import configparser

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

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
            Directory in which the config files is or will be created
        config_name : str
            Name of the config file
        help_message : str
            Message that describes how to edit/delete the config file using the CLI 
        """

        self._dashes = camus_log.get_log_dashes()

        self._config = configparser.ConfigParser()

        self._config_dir = config_dir
        self._config_path = f'{self._config_dir}/{config_name}'

        self.check_existence()

        if self._config_exists:
            self.read_config()

        self._help_message = help_message

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

                click.echo(f'Contents of the config file at {self._config_path}:')
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

    def _lammps_wizard(self):
        """
        A procedure to guide the user through LAMMPS setup.
        """

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

    def clean_config(self):
        """
        Deletes the file at ``self._config_path``.
        """

        camus_utils.delete_file(self._config_path)
