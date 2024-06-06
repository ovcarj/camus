import sys
import os
import click

import camus.utils.log as camus_log
import camus.utils.utils as camus_utils

from camus.db.db import DB
from camus.db.datadir import Datadir
from camus.cfg.config_camus import ConfigCamus
from camus.cfg.config_project import ConfigProject

from shutil import copyfile

class Project:
    """
    Class which handles creation, deletion, configuring, logging and querying projects.
    """

    def __init__(self, label=None, load_active=False):
        """
        If ``label`` is given, try to load the project data.
        Else, if ``load_active`` is True, load the currently active project.

        Parameters
        ----------
        label : str
            If given, the project data with the given label is loaded
        load_active : bool
            If True, load the currently active project

        """

        self._datadir = Datadir()
        self._db = DB()
        self._camus_cfg = ConfigCamus()

        self._db._get_projects()
        self._get_active_project()

        self.label = label

        self._check_existence()

        if self.label:
            self.load_project()

        elif load_active:
            self.load_project(label=self.active_project)

        self._dashes = camus_log.get_log_dashes()

    def load_project(self, label=None):
        """
        Search for the database entry of the project with the given ``label`` 
        and set the ``self.x`` attributes, where x = {label, dir, config, log, description}

        Parameters
        ----------
        label : None | str
            The project data with the given label is loaded. If ``None``, ``self.label`` is used

        """

        if label:
            self.label = label

        if self.label:

            self._check_existence()

            if self._proj_exists:

                proj_index = self._db._proj_labels.index(self.label)

                self.dir = self._db._proj_directories[proj_index]
                self.config = self._db._proj_configs[proj_index]
                self.log = self._db._proj_logs[proj_index]
                self.description = self._db._proj_descriptions[proj_index]

                self._get_active_batch()

                self._get_logger()

            else:
                click.echo(self._dashes)
                click.echo(f'The project with the label "{self.label}" does not exist.')
                click.echo(f'To see the list of existing projects, type: camus project list\n')

        else:
            click.echo('Cannot load project: no project label is provided.')

    def create_new_project(self, label, description=None, config_file=None):
        """
        Creates a new project with a given label.

        Parameters
        ----------
        label : str
            Label for the new project
        description : None | str
            Optional description of the project
        config_file : None | str
            Path to a project config file. If given, the file will be copied to the project directory

        """

        self.label = label
        self.dir = f'{self._datadir.projects}/{self.label}'
        self.config = f'{self.dir}/{self.label}.cfg'
        self.log = f'{self.dir}/{self.label}.log'
        self.description = description

        self._check_existence()

        if self._proj_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new project.')
            click.echo(f'Project directory with label "{self.label}" already exists at {self.dir}')
            click.echo(f'Project entry with label "{self.label}" already exists in the camus database')
            click.echo(self._dashes)
            click.echo(f'If you want to delete the existing project with the label "{self.label}", run:\n')
            click.echo(f'camus clean project {self.label}')
            click.echo(self._dashes)

        elif self._dir_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new project: project directory with label "{self.label}" already exists at {self.dir}')
            click.echo(self._dashes)
            click.echo(f'If you want to delete the existing project with the label "{self.label}", run:\n')
            click.echo(f'camus clean project {self.label}')
            click.echo(self._dashes)

            sys.exit()

        elif self._entry_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new project: project entry with label "{self.label}" already exists in the camus database\n')
            click.echo(f'If you want to delete the existing project entry with the label "{self.label}", run:\n')
            click.echo(f'camus clean project {self.label}')
            click.echo(self._dashes)

            sys.exit()

        else:

            click.echo(self._dashes)

            self._datadir._create_dir(dir_path=self.dir, dir_type='Project')

            self._db.create_new_project(
                    proj_label=self.label, 
                    proj_directory=self.dir,
                    proj_config=self.config,
                    proj_log=self.log,
                    proj_description=self.description
                    )

            self._check_existence()

            if self._proj_exists:

                click.echo(self._dashes)

                self._get_logger()

                self._write_2_log(camus_log.camus_start(start_message=f'Project {self.label} created on'))

                space_length = 15

                if self.description:
                    self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Description', self.description, space_length=space_length))

                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Directory', self.dir, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Config', self.config, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Log', self.log, space_length=space_length))

                self._write_2_log(self._dashes)

                if config_file:

                    click.echo(f'Using {config_file} for project configuration.')
                    click.echo(self._dashes)

                    copyfile(src=config_file, dst=self.config)

                else:
                    cfg = ConfigProject(self.config)
                    cfg.create_config_file()

                click.echo(f'To activate the created project, type:\n') 
                click.echo(f'camus project switch {self.label}')
                click.echo(self._dashes)
 
    def _get_active_project(self):
        """
        Read the main camus config file to get the active project.

        """

        self.active_project = self._camus_cfg._config['ActiveProject']['active_project']
    def _get_active_batch(self):
        """
        Read the project config file to get the active batch.

        """

        cfg = ConfigProject(self.config)
        self.active_batch = cfg._config['ActiveBatch']['active_batch']

    def _print_active_project(self):
        """
        Print information about the active project.

        """

        self._get_active_project()
        self.load_project(self.active_project)

        space_length = 12
        dashes = '-' * (len('Log') + len(self.log) + space_length - 2)

        click.echo(dashes)
        click.echo(f'Active project: {self.active_project}')

        if self.description:
            click.echo(f'{self.description}')

        click.echo(dashes)

        click.echo('{:<{space_length}} {:<{space_length}}'.format('Directory', self.dir, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Config', self.config, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Log', self.log, space_length=space_length))
        click.echo(dashes)

        if not self.active_batch:
            click.echo(f'No batch is activated.')
            click.echo(f'Use "camus batch switch <batch_label>" to activate a batch.')
            click.echo(dashes)

        else:
            click.echo(f'Active batch: {self.active_batch}')
            click.echo(dashes)

    def switch_active_project(self, label):
        """
        Edits the active_project entry in the main ``camus`` config file to ``project_label`` and loads the given project.

        Parameters
        ----------
        label : str
            Label of the project that will be activated.

        """

        self.label = label

        self.load_project()

        if self._proj_exists:

            update_dict = {
                    'ActiveProject': {'active_project': f'{self.label}'}
                    }

            self._camus_cfg.edit_config_file(update_dict)

            self._get_active_project()
            self._print_active_project()

        else:
            click.echo(self._dashes)
            click.echo(f'Could not activate project "{self.label}".')
            click.echo(self._dashes)

    def _goto_active_project(self):
        """
        Changes directory to the active project directory.
        """

        self._get_active_project()
        self.load_project(self.active_project)

        os.chdir(self.dir)

    def list_all_projects(self):
        """
        Lists all projects found in the ``projects`` table.

        """

        self._db._get_projects()
        self._get_active_project()

        if len(self._db._proj_labels) == 0:
            click.echo('No projects found. See `camus project new --help` on how to create new projects.')

        else:

            max_project_length = len('Label')
            max_description_length = len('Description')

            for project, description in zip(self._db._proj_labels, self._db._proj_descriptions):

                if len(project) > max_project_length:
                    max_project_length = len(project)

                if len(description) > max_description_length:
                    max_description_length = len(description)

            max_project_length += 8

            dashes = '-' * (max_project_length + max_description_length + 1)

            click.echo(dashes)
            click.echo('{:<{max_project_length}} {:<{max_project_length}}'.format('Label', 'Description', max_project_length=max_project_length))
            click.echo(dashes)

            for project, description in zip(self._db._proj_labels, self._db._proj_descriptions):

                if project == self.active_project:
                    project = '*' + project

                click.echo('{:<{max_project_length}} {:<{max_project_length}}'.format(project, description, max_project_length=max_project_length))

            click.echo(dashes)
            click.echo('(*) Currently active project')
            click.echo(dashes)

    def _proj_dir_exists(self):
        """
        Checks if ``{self._datadir.projects}/{self.label}`` exists. The self._dir_exists attribute is updated accordingly.

        """

        if self.label:

            _dir = f'{self._datadir.projects}/{self.label}'
            self._dir_exists = camus_utils.directory_exists(_dir)

        else:
            self._dir_exists = False

    def _proj_entry_exists(self):
        """
        Checks if ``self.label`` entry exists in the ``projects`` table of the camus database. The self._entry_exists attribute is updated accordingly.

        """

        self._db._get_projects()

        if self.label in self._db._proj_labels:
            self._entry_exists = True

        else:
            self._entry_exists = False

    def _check_existence(self):
        """
        Checks if both the project directory and project entry of ``self.label``  exist. The ``self._proj_exists`` attribute is updated accordingly.

        """

        self._proj_dir_exists()
        self._proj_entry_exists()

        if self._dir_exists and self._entry_exists:
            self._proj_exists = True

        else:
            self._proj_exists = False

    def _get_logger(self):
        """
        Gets the logger object from the ``self.log`` path

        """

        log_split = self.log.rpartition('/')
        logdir = log_split[0]
        logname = log_split[-1]

        self._logger = camus_log.init_logger(logdir=logdir, logname=logname)

    def _write_2_log(self, logtext):
        """
        Writes to the log file at the ``self.log`` path.

        Parameters
        ----------
        logtext : str
            Text to write in the log file

        """

        self._logger.info(logtext)

    def _print_log(self):
        """
        Prints the log file at ``self.log`` path.

        """

        with open(f'{self.log}', 'r') as f:
            lines = f.read()

        click.echo(lines)

    def _edit_active_config_by_subsection(self, subsection, value):
        """
        Edits the contents of the config file of the active project by passing only the subsection and the value, while the section is automatically found.

        Parameters
        ----------
        subsection : str
            Subsection in the config file
        value : str | float | int
            Value that the subsection is updated to

        """

        self._get_active_project()
        self.load_project(self.active_project)

        cfg = ConfigProject(self.config)
        cfg.edit_config_by_subsection(subsection=subsection, value=value)

    def _print_active_config(self):
        """
        Prints the contents of the config file of the active project.
        """

        self._get_active_project()
        self.load_project(self.active_project)

        cfg = ConfigProject(self.config)
        cfg.print_config()

    def delete_project(self, label):
        """
        Deletes the directory and database entry of the project with the ``label`` project label.

        """

        self.load_project(label=label)
        self._get_active_project()

        self._db._get_batches(self.label)

        click.echo(self._dashes)

        if self._proj_exists:

            click.echo(f'Deleting "{self.label}" project directory...')
            camus_utils.delete_directory(self.dir)

            click.echo(f'Deleting "{self.label}" project batch database entries...')

            for batch in self._db._batch_labels:
                self._db.clean_db_entry(table_name='batches', row_label=batch, column_name='batch_label')

            click.echo(f'Deleting "{self.label}" project database entry...')
            self._db.clean_db_entry(table_name='projects', row_label=f'{self.label}', column_name='proj_label')
            click.echo(self._dashes)

            self._check_existence()

            if not self._proj_exists:

                click.echo(f'Project "{self.label}" deleted.')
                click.echo(self._dashes)

                if self.label == self.active_project:

                    click.echo(f'Warning: deleted active project. To activate another project, run:\n')
                    click.echo(f'camus project switch <project_label>')
                    click.echo(self._dashes)

            else:
                click.echo(f'Failed to delete the project with label "{self.label}".')
                click.echo(self._dashes)

        else:
            click.echo(f'Cannot delete project with the label "{self.label}".')
            click.echo(self._dashes)
