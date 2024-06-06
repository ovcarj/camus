import os
import sys
import platformdirs
import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

from camus.cfg.config_camus import ConfigCamus

class Datadir:
    """
    Class which handles creation, deletion and checking of the
    ``camus`` data directory, which is found at ``platformdirs.user_data_dir(appname='camus')``.

    """

    def __init__(self):
        """
        Get the base directory from the config file.

        """

        config = ConfigCamus()

        self.base = config._config['camusDataDirectory']['data_directory']
        self.projects = f'{self.base}/projects'
        
        self._check_existence()

        self._dashes = camus_log.get_log_dashes()

    def _check_existence(self):
        """
        Checks the existence of the ``self.base`` directory; 
        the ``self._base_exists`` attribute is updated accordingly.

        """

        self._base_exists = camus_utils.directory_exists(self.base)

    def create_directories(self):
        """
        Creates ``camus`` data directories. If ``self.base`` already exists, 
        the user is prompted if they want to overwrite the directory.

        """

        input_ok = False
        
        if self._base_exists:

            while(input_ok == False):

                overwrite = input(f'WARNING: camus data base directory exists at {self.base}\nDo you want to overwrite it? All data will be deleted. [Y/n]\n')

                click.echo(self._dashes)

                try:
                    assert (overwrite == 'Y' or overwrite == 'n')
                    input_ok = True

                except AssertionError:
                    click.echo('Please enter "Y" or "n".')
                    click.echo(self._dashes)

            if overwrite == 'Y':

                self.clean_all_directories()

                click.echo(f'{self.base} deleted, as requested. Continuing...')
                click.echo(self._dashes)

            elif overwrite == 'n':

                click.echo('Will not overwrite the camus data directory.')
                click.echo(self._dashes)

                return

        os.makedirs(self.base)
        click.echo(f'camus base directory created at {self.base}')
        os.makedirs(f'{self.base}/projects')
        click.echo(f'Empty camus projects directory created at {self.base}/projects')

        self._check_existence()

        click.echo(self._dashes)

    def _create_dir(self, dir_path, dir_type):
        """
        Creates a directory ``dir_path``. Used to create ``dir_type = {'Project', 'Batch'}`` directories. #TODO probably also Calculation...

        Parameters
        ----------
        dir_path : str
            Path to the directory to be created
        dir_type : str
            Must be 'Project' or 'Batch'

        """

        if not ((dir_type == 'Project') or (dir_type == 'Batch')):
            click.echo(f'dir_type must be "Project" or "Batch". Exiting.')
            sys.exit()

        try:
            os.makedirs(name=dir_path, exist_ok=False)

            if camus_utils.directory_exists(dir_path):
                click.echo(f'{dir_type} directory created at {dir_path}')

            else:
                click.echo(f'Failed to create directory {dir_path}. Do you have the required permissions?')
                sys.exit()

        except OSError:
            click.echo(f'{dir_type} already exists at {dir_path}' )
            click.echo('Stopping.')
            sys.exit()

    def clean_all_directories(self):
        """
        Deletes all ``camus`` data directories.

        """

        self._check_existence()

        if not self._base_exists:
            click.echo('Nothing to clean.')

        else:
            camus_utils.delete_directory(self.base)

        self._check_existence()
