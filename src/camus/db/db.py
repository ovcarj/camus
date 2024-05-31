import sys
import click

import sqlite3

import camus.utils.log as camus_log
import camus.utils.utils as camus_utils

from camus.db.datadir import Datadir

class DB:
    """
    Class which handles creation, deletion and updating of the ``camus`` database,
    which is found at ``platformdirs.user_data_dir(appname='camus')/camus.db``
    """

    def __init__(self):
        """
        If the database exists, establishes a connection and a cursor.
        """

        datadir = Datadir()

        self._base = datadir.base
        
        self._db_path = f'{self._base}/camus.db'
        self.check_existence()

        if self._db_exists:

            self._con = sqlite3.connect(self._db_path)
            self._cur = self._con.cursor()

        self._dashes = camus_log.get_log_dashes()

    def check_existence(self):
        """
        Checks the existence of the ``camus`` database;
        the ``self._db_exists`` attribute is updated accordingly
        """

        self._db_exists = camus_utils.file_exists(self._db_path)

    def create_database(self):
        """
        If the camus database doesn't exist, creates it.
        """

        self.check_existence()

        if self._db_exists:

            click.echo(f'WARNING: camus database already exists at {self._db_path}')
            click.echo(self._dashes)
            self._handle_database_exists()

        else:

            try:
                self._con = sqlite3.connect(self._db_path)
                self._cur = self._con.cursor()

            except:
                click.echo(f'Failed to initialize the camus database at {self._db_path}. Do you have the required permissions to write to the requested directory? Exiting.')
                sys.exit()

            self.check_existence()

            if self._db_exists:

                self._cur.execute('CREATE TABLE projects(proj_label, proj_directory, proj_config, proj_log)')
                self._cur.execute('CREATE TABLE batches(proj_label, batch_label, batch_directory, batch_config, batch_log)')
                self._cur.execute('CREATE TABLE calculations(proj_label, batch_label, calc_label, calc_directory, calc_log)')
                self._con.commit()

                click.echo(f'camus database initialized at {self._db_path}')

            else:
                click.echo(f'Failed to initialize the camus database at {self._db_path}. Exiting.')
                sys.exit()

    def _handle_database_exists(self):
        """
        Handles the case when the user tries to run ``self.create_database()`` with a preexisting database.
        """
        
        input_ok = False

        while(input_ok == False):

            proceed = input(f'Do you want to proceed with the current database? [Y/n]\n')

            click.echo(self._dashes)

            try:
                assert (proceed == 'Y' or proceed == 'n')
                input_ok = True

            except AssertionError:
                click.echo('Please enter "Y" or "n".')
                click.echo(self._dashes)

        if proceed == 'Y':

            click.echo('Proceeding with the current camus database.')
            click.echo(self._dashes)

        elif proceed == 'n':

            click.echo(f'Stopping, as requested.')
            click.echo(f'Run "camus clean db" to delete the current database before proceeding.')
            click.echo(self._dashes)

            sys.exit()

    def clean_database(self):
        """
        Deletes the database found at ``self._db_path``.
        """

        self.check_existence()

        if not self._db_exists:
            click.echo(f'camus database at {self._db_path} does not exist.')

        else:
            camus_utils.delete_file(self._db_path)
