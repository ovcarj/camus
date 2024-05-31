import sys
import click

import camus.utils.log as camus_log
import camus.utils.utils as camus_utils

from camus.db.db import DB
from camus.db.datadir import Datadir

class Project:
    """
    Class which handles creation, deletion, configuring, logging and querying projects.
    """

    def __init__(self, label=None):
        """
        If ``label`` is given, load the project data.

        Parameters
        ----------
        label : str
            If given, the project data with the given label is loaded

        """

        self._datadir = Datadir()
        self._db = DB()

        self.label = label

#        if self.label:
#            self._proj_dir_exists()

        self._dashes = camus_log.get_log_dashes()

    def create_new_project(self, label, description=None):
        """
        Creates a new project with a given label.

        Parameters
        ----------
        label : str
            Label for the new project
        description : None | str
            Optional description of the project

        """

        self.label = label
        self.dir = f'{self._datadir.projects}/{self.label}'
        self.config = f'{self.dir}/{self.label}.cfg'
        self.log = f'{self.dir}/{self.label}.log'
        self.description = description

        self._proj_dir_exists()
#        self._project_db_entry_exists()

        if self._dir_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new project: project directory with label "{self.label}" already exists at {self.dir}\n')
            click.echo(f'If you want to delete the existing project with the label "{self.label}", run:\n')
            click.echo(f'camus project switch {self.label}')
            click.echo(f'camus clean project')
            click.echo(self._dashes)

            sys.exit()

#        elif self._db_entry_exists:

        else:

            self._datadir.create_project_dir(project_dir=self.dir)
            self._proj_dir_exists()

            self._db.create_new_project(
                    proj_label=self.label, 
                    proj_directory=self.dir,
                    proj_config=self.config,
                    proj_log=self.log,
                    proj_description=self.description
                    )

    def _proj_dir_exists(self):
        """
        Checks if ``{self._datadir.projects}/{project_label}`` exists. The self._dir_exists attribute is updated accordingly.

        Parameters
        ----------
        project_label : str
            Label of the project whose directory is checked for existence

        """

        if self.label:
            self._dir_exists = camus_utils.directory_exists(self.dir)

        else:
            self._dir_exists = False
