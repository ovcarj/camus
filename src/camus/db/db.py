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
        self._check_existence()

        if self._db_exists:

            self._con = sqlite3.connect(self._db_path)
            self._cur = self._con.cursor()

        self._dashes = camus_log.get_log_dashes()

    def _check_existence(self):
        """
        Checks the existence of the ``camus`` database;
        the ``self._db_exists`` attribute is updated accordingly

        """

        self._db_exists = camus_utils.file_exists(self._db_path)

    def create_database(self):
        """
        Creates the camus database.

        """

        self._check_existence()

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

            self._check_existence()

            if self._db_exists:

                self._cur.execute('CREATE TABLE projects(proj_label, proj_directory, proj_config, proj_log, proj_description)')
                self._cur.execute('CREATE TABLE batches(proj_label, batch_label, batch_directory, batch_config, batch_log, batch_description)')
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

        proceed = camus_log.ask_yes_no(f'Do you want to proceed with the current database? [Y/n]\n')

        if proceed == 'Y':

            click.echo('Proceeding with the current camus database.')

        elif proceed == 'n':

            click.echo(f'Stopping, as requested.')
            click.echo(f'Run "camus clean db" to delete the current database before proceeding.')
            click.echo(self._dashes)

            sys.exit()

    def create_new_project(self, proj_label, proj_directory, proj_config, proj_log, proj_description=None):
        """
        Updates the ``projects`` table with the new project data.

        Parameters
        ----------
        proj_label : str
            Label for the new project
        proj_directory : str
            Path to the project directory
        proj_config : str
            Path to the project configuration file
        proj_log : str
            Path to the project log file
        proj_description : None
            Optional project description

        """

        if not proj_description:
            proj_description = ''

        data = ({
            'proj_label': proj_label,
            'proj_directory': proj_directory,
            'proj_config': proj_config,
            'proj_log': proj_log,
            'proj_description': proj_description
                })

        self._cur.execute("""
        INSERT INTO projects VALUES(:proj_label, :proj_directory, :proj_config, :proj_log, :proj_description) 
        """, data)

        self._con.commit()

        click.echo(f'Project entry "{proj_label}" added to the camus database.')

    def _get_projects(self):
        """
        Stores project data to ``self._x``, where x = {proj_labels, proj_directories, proj_configs, proj_logs, proj_descriptions}.

        """

        projects_data = self._cur.execute("""
        SELECT proj_label, proj_directory, proj_config, proj_log, proj_description FROM projects ORDER BY proj_label ASC
        """).fetchall()

        self._proj_labels = []
        self._proj_directories = []
        self._proj_configs = []
        self._proj_logs = []
        self._proj_descriptions = []

        for project_data in projects_data:

            self._proj_labels.append(project_data[0])
            self._proj_directories.append(project_data[1])
            self._proj_configs.append(project_data[2])
            self._proj_logs.append(project_data[3])
            self._proj_descriptions.append(project_data[4])

    def clean_database(self):
        """
        Deletes the database found at ``self._db_path``.

        """

        self._check_existence()

        if not self._db_exists:
            click.echo(f'camus database at {self._db_path} does not exist.')

        else:
            camus_utils.delete_file(self._db_path)

    def clean_db_entry(self, table_name, row_label):
        """
        Deletes the row that starts with ``row_label`` from the table ``table_name``.

        Parameters
        ----------
        table_name : str
            Name of the table from which a row is being deleted
        row_label : str
            Value of the first entry of the row that is being deleted
        
        """

        column_name = self._cur.execute(f'PRAGMA table_info({table_name})').fetchall()[0][1]

        data = ({
            'row_label': row_label,
            })

        entry = self._cur.execute(f"""
        SELECT {column_name} FROM {table_name} WHERE {column_name}=:row_label
        """, data).fetchall()

        if entry:

            self._cur.execute(f"""
            DELETE FROM {table_name} WHERE {column_name}=:row_label 
            """, data)

            self._con.commit()

            entry = self._cur.execute(f"""
            SELECT {column_name} FROM {table_name} WHERE {column_name}=:row_label
            """, data).fetchall()

            if not entry:
                click.echo(f'Deleted "{row_label}" database entry from the "{table_name}" table.')
            else:
                click.echo(f'Failed to delete "{row_label}" database entry from the "{table_name}" table.')

        else:
            click.echo(f'Database entry "{row_label}" from the "{table_name}" table does not exist.')
