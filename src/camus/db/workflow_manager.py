import sys
import os
import click

import camus.utils.log as camus_log
import camus.utils.utils as camus_utils

from camus.db.db import DB
from camus.db.project_manager import ProjectManager
from camus.db.datadir import Datadir

from camus.cfg.config_project import ConfigProject
from camus.cfg.config_workflow import ConfigWorkflow

from camus.workflows.search_workflows import get_wf_class_by_tag

from shutil import copyfile

class WorkflowManager:
    """
    Class which handles creation, deletion, configuring, logging and querying workflows.

    """

    def __init__(self, proj_label=None, workflow_label=None, load_active=False):
        """
        If the workflow exists, loads the workflow with ``workflow_label`` belonging to project ``proj_label``.

        If ``proj_label`` is not given, the currently active project is used.
        If ``workflow_label`` is given, try to load the workflow data with the given ``workflow_label``.
        Else, if ``load_active`` is True, load the currently active workflow.

        Parameters
        ----------
        proj_label : None | str
            Label of the project that the workflow belongs to. If ``None``, load the currently active project.
        workflow_label : str
            If given, the workflow data with the given label is loaded
        load_active : bool
            If True, load the currently active workflow

        """

        self._dashes = camus_log.get_log_dashes()

        self._datadir = Datadir()
        self._db = DB()

        self._db._get_projects()

        if not proj_label:
            self._project = ProjectManager(load_active=True)

        else:
            self._project = ProjectManager(label=proj_label)

        self._proj_cfg = ConfigProject(self._project.config)

        self.label = workflow_label
        self._get_active_workflow()

        self._check_existence()

        if self.label:
            self.load_workflow()

        elif load_active:
            self.load_workflow(label=self.active_workflow)

    def load_workflow(self, label=None):
        """
        Search for the database entry of the workflow with the given ``label`` 
        and set the ``self.x`` attributes, where x = {label, dir, config, log, description}.

        Read the ``[WORKFLOW]['workflow_tag']`` from the config file and instantiate the corresponding workflow class.

        Parameters
        ----------
        label : None | str
            The workflow data with the given label is loaded. If ``None``, ``self.label`` is used

        """

        if label:
            self.label = label

        if self.label:

            self._check_existence()

            if self._workflow_exists:

                workflow_index = self._db._workflow_labels.index(self.label)

                self.dir = self._db._workflow_directories[workflow_index]
                self.config_filepath = self._db._workflow_configs[workflow_index]
                self.log = self._db._workflow_logs[workflow_index]
                self.description = self._db._workflow_descriptions[workflow_index]

                self.config = ConfigWorkflow(self.config_filepath)

                try:
                    self._workflow_cls = get_wf_class_by_tag(self.config._config['WORKFLOW']['workflow_tag'])

                except:
                    pass

            else:
                click.echo(self._dashes)
                click.echo(f'The workflow "{self.label}" does not exist.')
                click.echo(f'To see the list of existing workflows, type: camus wf list\n')

        else:
            click.echo('Cannot load workflow: no workflow label is provided.')

    def create_new_workflow(self, label, description=None, config_file=None):
        """
        Creates a new workflow with a given label.

        Parameters
        ----------
        label : str
            Label for the new workflow
        description : None | str
            Optional description of the workflow
        config_file : None | str
            Path to a workflow config file. If given, the file will be copied to the workflow directory

        """

        self.label = label
        self.dir = f'{self._project.dir}/{self.label}'
        self.config_filepath = f'{self.dir}/{self.label}.cfg'
        self.log = f'{self.dir}/{self.label}.log'
        self.description = description

        self._check_existence()

        if self._workflow_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new workflow.')
            click.echo(f'Workflow directory with label "{self.label}" already exists at {self.dir}')
            click.echo(f'Workflow entry with label "{self.label}" already exists in the camus database')
            click.echo(self._dashes)
            click.echo(f'If you want to delete the existing workflow with the label "{self.label}", run:\n')
            click.echo(f'camus clean wf {self.label}')
            click.echo(self._dashes)

        elif self._dir_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new workflow: workflow directory with label "{self.label}" already exists at {self.dir}')
            click.echo(self._dashes)
            click.echo(f'If you want to delete the existing workflow with the label "{self.label}", run:\n')
            click.echo(f'camus clean workflow {self.label}')
            click.echo(self._dashes)

            sys.exit()

        elif self._entry_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new workflow: workflow entry with label "{self.label}" already exists in the camus database\n')
            click.echo(f'If you want to delete the existing workflow entry with the label "{self.label}", run:\n')
            click.echo(f'camus clean workflow {self.label}')
            click.echo(self._dashes)

            sys.exit()

        else:

            click.echo(self._dashes)

            self._datadir._create_dir(dir_path=self.dir, dir_type='Workflow')

            self._db.create_new_workflow(
                    proj_label=self._project.label,
                    workflow_label=self.label, 
                    workflow_directory=self.dir,
                    workflow_config=self.config_filepath,
                    workflow_log=self.log,
                    workflow_description=self.description
                    )

            self._check_existence()

            if self._workflow_exists:

                click.echo(self._dashes)

                self._project._write_2_log(camus_log.timestamp_message(f'Workflow {self.label} created on'))
                self._project._write_2_log(self._dashes)
                self._write_2_log(camus_log.camus_start(start_message=f'Workflow {self.label} created on'))

                space_length = 15

                if self.description:
                    self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Description', self.description, space_length=space_length))

                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Directory', self.dir, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Config', self.config_filepath, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Log', self.log, space_length=space_length))

                self._write_2_log(self._dashes)

                if config_file:

                    click.echo(f'Using {config_file} for workflow configuration.')
                    click.echo(self._dashes)

                    copyfile(src=config_file, dst=self.config_filepath)

                else:

                    cfg = ConfigWorkflow(self.config_filepath)
                    cfg.create_config_file()

                self.load_workflow()

                self._write_2_log('Workflow class info:\n')

                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Module', self._workflow_cls.__module__, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Class', self._workflow_cls.__name__, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Tag', self._workflow_cls.workflow_tag, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Description', self._workflow_cls.workflow_description, space_length=space_length))

                self._write_2_log(self._dashes)

                click.echo(f'To activate the created workflow, type:\n') 
                click.echo(f'camus wf switch {self.label}')
                click.echo(self._dashes)
 
    def _get_active_workflow(self):
        """
        Read the project config file to get the active workflow.

        """

        self._project._get_active_workflow()
        self.active_workflow = self._project.active_workflow

    def _print_active_workflow(self):
        """
        Print information about the active workflow.

        """

        self._get_active_workflow()
        self.load_workflow(self.active_workflow)

        space_length = 12
        dashes = '-' * (len('Log') + len(self.log) + space_length - 2)

        click.echo(dashes)
        click.echo(f'Project: {self._project.active_project}')
        click.echo(dashes)
        click.echo(f'Active workflow: {self.label}')

        if self.description:
            click.echo(f'{self.description}')

        click.echo(dashes)

        click.echo('{:<{space_length}} {:<{space_length}}'.format('Directory', self.dir, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Config', self.config_filepath, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Log', self.log, space_length=space_length))
        click.echo(dashes)

        click.echo('{:<{space_length}} {:<{space_length}}'.format('Module', self._workflow_cls.__module__, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Class', self._workflow_cls.__name__, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Tag', self._workflow_cls.workflow_tag, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Description', self._workflow_cls.workflow_description, space_length=space_length))
        click.echo(dashes)

    def switch_active_workflow(self, label):
        """
        Edits the active_workflow entry in the project config file to ``workflow_label`` and loads the given workflow.

        Parameters
        ----------
        label : str
            Label of the workflow that will be activated.

        """

        self.label = label

        self.load_workflow()

        if self._workflow_exists:

            update_dict = {
                    'ActiveWorkflow': {'active_workflow': f'{self.label}'}
                    }

            self._proj_cfg.edit_config_file(update_dict)

            self._get_active_workflow()
            self._print_active_workflow()

        else:
            click.echo(self._dashes)
            click.echo(f'Could not activate workflow "{self.label}".')
            click.echo(self._dashes)

    def _goto_active_workflow(self):
        """
        Changes directory to the active workflow directory.
        """

        self._get_active_workflow()
        self.load_workflow(self.active_workflow)

        os.chdir(self.dir)

    def list_all_workflows(self):
        """
        Lists all workflows found in the ``workflows`` table with ``proj_label==self._project.label``.

        """

        self._db._get_workflows(proj_label=self._project.label)
        self._get_active_workflow()

        if len(self._db._workflow_labels) == 0:
            click.echo('No workflows found. See `camus wf new --help` on how to create new projects.')

        else:

            max_workflow_length = len('Label')
            max_description_length = len('Description')

            for workflow, description in zip(self._db._workflow_labels, self._db._workflow_descriptions):

                if len(workflow) > max_workflow_length:
                    max_workflow_length = len(workflow)

                if len(description) > max_description_length:
                    max_description_length = len(description)

            max_workflow_length += 8

            dashes = '-' * (max_workflow_length + max_description_length + 1)

            click.echo(dashes)
            click.echo('{:<{max_workflow_length}} {:<{max_workflow_length}}'.format('Label', 'Description', max_workflow_length=max_workflow_length))
            click.echo(dashes)

            for workflow, description in zip(self._db._workflow_labels, self._db._workflow_descriptions):

                if workflow == self.active_workflow:
                    workflow = '*' + workflow

                click.echo('{:<{max_workflow_length}} {:<{max_workflow_length}}'.format(workflow, description, max_workflow_length=max_workflow_length))

            click.echo(dashes)
            click.echo('(*) Currently active workflow')
            click.echo(dashes)

    def _workflow_dir_exists(self):
        """
        Checks if ``{self._datadir.projects}/{self._project.label}/{self.label}`` exists. The self._dir_exists attribute is updated accordingly.

        """

        if self.label:

            _dir = f'{self._datadir.projects}/{self._project.label}/{self.label}'
            self._dir_exists = camus_utils.directory_exists(_dir)

        else:
            self._dir_exists = False

    def _workflow_entry_exists(self):
        """
        Checks if ``self.label`` entry exists in the ``workflows`` table of the camus database. The self._entry_exists attribute is updated accordingly.

        """

        self._db._get_workflows(proj_label=self._project.label)

        if self.label in self._db._workflow_labels:
            self._entry_exists = True

        else:
            self._entry_exists = False

    def _check_existence(self):
        """
        Checks if both the workflow directory and workflow entry of ``self.label``  exist. The ``self._workflow_exists`` attribute is updated accordingly.

        """

        self._workflow_dir_exists()
        self._workflow_entry_exists()

        if self._dir_exists and self._entry_exists:
            self._workflow_exists = True

        else:
            self._workflow_exists = False

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

        if not hasattr(self, '_logger'):
            self._get_logger()

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
        Edits the contents of the config file of the active workflow by passing only the subsection and the value, while the section is automatically found.

        Parameters
        ----------
        subsection : str
            Subsection in the config file
        value : str | float | int
            Value that the subsection is updated to

        """

        self._get_active_workflow()
        self.load_workflow(self.active_workflow)

        cfg = ConfigWorkflow(self.config_filepath)
        cfg.edit_config_by_subsection(subsection=subsection, value=value)

    def _print_active_config(self):
        """
        Prints the contents of the config file of the active workflow.
        """

        self._get_active_workflow()
        self.load_workflow(self.active_workflow)

        cfg = ConfigWorkflow(self.config_filepath)
        cfg.print_config()

    def delete_workflow(self, label):
        """
        Deletes the directory and database entry of the workflow with the ``label`` workflow label.

        """

        self.load_workflow(label=label)
        self._get_active_workflow()

        click.echo(self._dashes)

        if self._workflow_exists:

            click.echo(f'Deleting "{self.label}" workflow directory...')
            camus_utils.delete_directory(self.dir)

            click.echo(f'Deleting "{self.label}" workflow database entry...')
            self._db.clean_db_entry(table_name='workflows', row_label=f'{self.label}', column_name='workflow_label')
            click.echo(self._dashes)

            self._check_existence()

            if not self._workflow_exists:

                click.echo(f'Workflow "{self.label}" deleted.')
                click.echo(self._dashes)

                self._project._write_2_log(camus_log.timestamp_message(f'Workflow {self.label} deleted on'))
                self._project._write_2_log(self._dashes)

                if self.label == self.active_workflow:

                    click.echo(f'Warning: deleted active workflow. To activate another workflow, run:\n')
                    click.echo(f'camus wf switch <workflow_label>')
                    click.echo(self._dashes)

            else:
                click.echo(f'Failed to delete the workflow with label "{self.label}".')
                click.echo(self._dashes)

        else:
            click.echo(f'Cannot delete workflow with the label "{self.label}".')
            click.echo(self._dashes)
