import sys
import os
import click

import camus.utils.log as camus_log
import camus.utils.utils as camus_utils

from camus.db.db import DB
from camus.db.project import Project
from camus.db.datadir import Datadir
from camus.cfg.config_project import ConfigProject
from camus.cfg.config_batch import ConfigBatch

from shutil import copyfile

class Batch:
    """
    Class which handles creation, deletion, configuring, logging and querying batches.

    """

    def __init__(self, proj_label=None, batch_label=None, load_active=False):
        """
        If the batch exists, loads the batch with ``batch_label`` belonging to project ``proj_label``.

        If ``proj_label`` is not given, the currently active project is used.
        If ``batch_label`` is given, try to load the batch data with the given ``batch_label``.
        Else, if ``load_active`` is True, load the currently active batch.

        Parameters
        ----------
        proj_label : None | str
            Label of the project that the batch belongs to. If ``None``, load the currently active project.
        batch_label : str
            If given, the batch data with the given label is loaded
        load_active : bool
            If True, load the currently active batch

        """

        self._datadir = Datadir()
        self._db = DB()

        self._db._get_projects()

        if not proj_label:
            self._project = Project(load_active=True)

        else:
            self._project = Project(label=proj_label)

        self._proj_cfg = ConfigProject(self._project.config)

        self.label = batch_label
        self._get_active_batch()

        self._check_existence()

        if self.label:
            self.load_batch()

        elif load_active:
            self.load_batch(label=self.active_batch)

        self._dashes = camus_log.get_log_dashes()

    def load_batch(self, label=None):
        """
        Search for the database entry of the batch with the given ``label`` 
        and set the self.x attributes, where x = {label, dir, config, log, description}

        Parameters
        ----------
        label : None | str
            The batch data with the given label is loaded. If ``None``, ``self.label`` is used

        """

        if label:
            self.label = label

        if self.label:

            self._check_existence()

            if self._batch_exists:

                batch_index = self._db._batch_labels.index(self.label)

                self.dir = self._db._batch_directories[batch_index]
                self.config = self._db._batch_configs[batch_index]
                self.log = self._db._batch_logs[batch_index]
                self.description = self._db._batch_descriptions[batch_index]

                self._get_logger()

            else:
                click.echo(self._dashes)
                click.echo(f'The batch "{self.label}" does not exist.')
                click.echo(f'To see the list of existing batches, type: camus batch list\n')

        else:
            click.echo('Cannot load batch: no batch label is provided.')

    def create_new_batch(self, label, description=None, config_file=None):
        """
        Creates a new batch with a given label.

        Parameters
        ----------
        label : str
            Label for the new batch
        description : None | str
            Optional description of the batch
        config_file : None | str
            Path to a batch config file. If given, the file will be copied to the batch directory

        """

        self.label = label
        self.dir = f'{self._project.dir}/{self.label}'
        self.config = f'{self.dir}/{self.label}.cfg'
        self.log = f'{self.dir}/{self.label}.log'
        self.description = description

        self._check_existence()

        if self._batch_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new batch.')
            click.echo(f'Batch directory with label "{self.label}" already exists at {self.dir}')
            click.echo(f'Batch entry with label "{self.label}" already exists in the camus database')
            click.echo(self._dashes)
            click.echo(f'If you want to delete the existing batch with the label "{self.label}", run:\n')
            click.echo(f'camus clean batch {self.label}')
            click.echo(self._dashes)

        elif self._dir_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new batch: batch directory with label "{self.label}" already exists at {self.dir}')
            click.echo(self._dashes)
            click.echo(f'If you want to delete the existing batch with the label "{self.label}", run:\n')
            click.echo(f'camus clean batch {self.label}')
            click.echo(self._dashes)

            sys.exit()

        elif self._entry_exists:

            click.echo(self._dashes)
            click.echo(f'Cannot create new batch: batch entry with label "{self.label}" already exists in the camus database\n')
            click.echo(f'If you want to delete the existing batch entry with the label "{self.label}", run:\n')
            click.echo(f'camus clean batch {self.label}')
            click.echo(self._dashes)

            sys.exit()

        else:

            click.echo(self._dashes)

            self._datadir._create_dir(dir_path=self.dir, dir_type='Batch')

            self._db.create_new_batch(
                    proj_label=self._project.label,
                    batch_label=self.label, 
                    batch_directory=self.dir,
                    batch_config=self.config,
                    batch_log=self.log,
                    batch_description=self.description
                    )

            self._check_existence()

            if self._batch_exists:

                click.echo(self._dashes)

                self._get_logger()

                self._project._write_2_log(camus_log.timestamp_message(f'Batch {self.label} created on'))
                self._project._write_2_log(self._dashes)
                self._write_2_log(camus_log.camus_start(start_message=f'Batch {self.label} created on'))

                space_length = 15

                if self.description:
                    self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Description', self.description, space_length=space_length))

                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Directory', self.dir, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Config', self.config, space_length=space_length))
                self._write_2_log('{:<{space_length}} {:<{space_length}}'.format('Log', self.log, space_length=space_length))

                self._write_2_log(self._dashes)

                if config_file:

                    click.echo(f'Using {config_file} for batch configuration.')
                    click.echo(self._dashes)

                    copyfile(src=config_file, dst=self.config)

                else:
                    cfg = ConfigBatch(self.config)
                    cfg.create_config_file()

                click.echo(f'To activate the created batch, type:\n') 
                click.echo(f'camus batch switch {self.label}')
                click.echo(self._dashes)
 
    def _get_active_batch(self):
        """
        Read the project config file to get the active batch.

        """

        self._project._get_active_batch()
        self.active_batch = self._project.active_batch

    def _print_active_batch(self):
        """
        Print information about the active batch.

        """

        self._get_active_batch()
        self.load_batch(self.active_batch)

        space_length = 12
        dashes = '-' * (len('Log') + len(self.log) + space_length - 2)

        click.echo(dashes)
        click.echo(f'Project: {self._project.active_project}')
        click.echo(dashes)
        click.echo(f'Active batch: {self.label}')

        if self.description:
            click.echo(f'{self.description}')

        click.echo(dashes)

        click.echo('{:<{space_length}} {:<{space_length}}'.format('Directory', self.dir, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Config', self.config, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Log', self.log, space_length=space_length))
        click.echo(dashes)

    def switch_active_batch(self, label):
        """
        Edits the active_batch entry in the project config file to ``batch_label`` and loads the given batch.

        Parameters
        ----------
        label : str
            Label of the batch that will be activated.

        """

        self.label = label

        self.load_batch()

        if self._batch_exists:

            update_dict = {
                    'ActiveBatch': {'active_batch': f'{self.label}'}
                    }

            self._proj_cfg.edit_config_file(update_dict)

            self._get_active_batch()
            self._print_active_batch()

        else:
            click.echo(self._dashes)
            click.echo(f'Could not activate batch "{self.label}".')
            click.echo(self._dashes)

    def _goto_active_batch(self):
        """
        Changes directory to the active batch directory.
        """

        self._get_active_batch()
        self.load_batch(self.active_batch)

        os.chdir(self.dir)

    def list_all_batches(self):
        """
        Lists all batches found in the ``batches`` table with ``proj_label==self._project.label``.

        """

        self._db._get_batches(proj_label=self._project.label)
        self._get_active_batch()

        if len(self._db._batch_labels) == 0:
            click.echo('No batches found. See `camus batch new --help` on how to create new projects.')

        else:

            max_batch_length = len('Label')
            max_description_length = len('Description')

            for batch, description in zip(self._db._batch_labels, self._db._batch_descriptions):

                if len(batch) > max_batch_length:
                    max_batch_length = len(batch)

                if len(description) > max_description_length:
                    max_description_length = len(description)

            max_batch_length += 8

            dashes = '-' * (max_batch_length + max_description_length + 1)

            click.echo(dashes)
            click.echo('{:<{max_batch_length}} {:<{max_batch_length}}'.format('Label', 'Description', max_batch_length=max_batch_length))
            click.echo(dashes)

            for batch, description in zip(self._db._batch_labels, self._db._batch_descriptions):

                if batch == self.active_batch:
                    batch = '*' + batch

                click.echo('{:<{max_batch_length}} {:<{max_batch_length}}'.format(batch, description, max_batch_length=max_batch_length))

            click.echo(dashes)
            click.echo('(*) Currently active batch')
            click.echo(dashes)

    def _batch_dir_exists(self):
        """
        Checks if ``{self._datadir.projects}/{self._project.label}/{self.label}`` exists. The self._dir_exists attribute is updated accordingly.

        """

        if self.label:

            _dir = f'{self._datadir.projects}/{self._project.label}/{self.label}'
            self._dir_exists = camus_utils.directory_exists(_dir)

        else:
            self._dir_exists = False

    def _batch_entry_exists(self):
        """
        Checks if ``self.label`` entry exists in the ``batches`` table of the camus database. The self._entry_exists attribute is updated accordingly.

        """

        self._db._get_batches(proj_label=self._project.label)

        if self.label in self._db._batch_labels:
            self._entry_exists = True

        else:
            self._entry_exists = False

    def _check_existence(self):
        """
        Checks if both the batch directory and batch entry of ``self.label``  exist. The ``self._batch_exists`` attribute is updated accordingly.

        """

        self._batch_dir_exists()
        self._batch_entry_exists()

        if self._dir_exists and self._entry_exists:
            self._batch_exists = True

        else:
            self._batch_exists = False

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
        Edits the contents of the config file of the active batch by passing only the subsection and the value, while the section is automatically found.

        Parameters
        ----------
        subsection : str
            Subsection in the config file
        value : str | float | int
            Value that the subsection is updated to

        """

        self._get_active_batch()
        self.load_batch(self.active_batch)

        cfg = ConfigBatch(self.config)
        cfg.edit_config_by_subsection(subsection=subsection, value=value)

    def _print_active_config(self):
        """
        Prints the contents of the config file of the active batch.
        """

        self._get_active_batch()
        self.load_batch(self.active_batch)

        cfg = ConfigBatch(self.config)
        cfg.print_config()

    def delete_batch(self, label):
        """
        Deletes the directory and database entry of the batch with the ``label`` batch label.

        """

        self.load_batch(label=label)
        self._get_active_batch()

        click.echo(self._dashes)

        if self._batch_exists:

            click.echo(f'Deleting "{self.label}" batch directory...')
            camus_utils.delete_directory(self.dir)

            click.echo(f'Deleting "{self.label}" batch database entry...')
            self._db.clean_db_entry(table_name='batches', row_label=f'{self.label}', column_name='batch_label')
            click.echo(self._dashes)

            self._check_existence()

            if not self._batch_exists:

                click.echo(f'Batch "{self.label}" deleted.')
                click.echo(self._dashes)

                self._project._write_2_log(camus_log.timestamp_message(f'Batch {self.label} deleted on'))
                self._project._write_2_log(self._dashes)

                if self.label == self.active_batch:

                    click.echo(f'Warning: deleted active batch. To activate another batch, run:\n')
                    click.echo(f'camus batch switch <batch_label>')
                    click.echo(self._dashes)

            else:
                click.echo(f'Failed to delete the batch with label "{self.label}".')
                click.echo(self._dashes)

        else:
            click.echo(f'Cannot delete batch with the label "{self.label}".')
            click.echo(self._dashes)
