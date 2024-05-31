import click

import camus.utils.log as camus_log
import camus.utils.utils as camus_utils

from camus.db.db import DB
from camus.cfg.datadir import Datadir

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

        if self.label:
            self._project_dir_exists()

    def create_new_project(self, label):
        """
        Creates a new project with a given label.

        Parameters
        ----------
        label : str
            Label for the new project

        """
        


    def _project_dir_exists(self):
        """
        Checks if ``{self.base}/projects/{project_label}`` exists. The self._dir_exists attribute is updated accordingly.

        Parameters
        ----------
        project_label : str
            Label of the project whose directory is checked for existence

        """

        if self.label:
            self._dir_exists = camus_utils.directory_exists

        else:
            self._dir_exists = False
