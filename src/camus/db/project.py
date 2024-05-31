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

        datadir = Datadir()
        self._base = datadir.base
        self._projects_base = f'{self._base}/projects'

        self.label = label

        self._db = DB()

        if self.label is not None:
            # Load project here
            pass

    def new_project(self, label):
        """
        Creates a new project with a given label.

        Parameters
        ----------
        label : str
            Label for the new project

        """
        pass 

