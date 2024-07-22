import sys
import click

import camus.utils.log as camus_log

from camus.cfg.config import Config
from camus.cfg.config_project import ConfigProject

from camus.db.project_manager import ProjectManager

from camus.workflows import search_workflows

class ConfigWorkflow(Config):
    """
    Class which handles creation, deletion, reading and editing of the 
    workflow config files.

    """

    def __init__(self, workflow_config_path):
        """
        Check for the existence of the config file. If it exists, read its contents.

        Parameters
        ----------
        workflow_config_path : str
            Path to the workflow config file

        """

        workflow_cfg_split = workflow_config_path.rpartition('/')

        config_dir = workflow_cfg_split[0]
        config_name = workflow_cfg_split[-1]

        super().__init__(config_dir=config_dir, config_name=config_name,  
                help_message='Check "camus wf config edit --help" for instructions on editing the workflow config file.')

    def _define_calc_options_messages(self):
        """
        Define a dictionary ``self._calc_options_messages`` of messages to print
        for various calculation options.

        """

        self._calc_options_messages = {
                'structures_file': 'Please enter a path to an ASE-readable structure(s) file.',
                'lammps_input_file': 'Please enter a path to a LAMMPS input file.'
                }

    def define_default_values(self):
        """
        Defines the default values for a workflow config file. The majority of the defaults are taken from the active project config file.

        """

        default_energy_force_engine = ''
        default_calculation_type = ''
        default_path_to_structures = ''

        self._config['WORKFLOW'] = {}

        proj = ProjectManager(load_active=True)
        proj_cfg = ConfigProject(proj.config)

        self._config['LAMMPS_SETUP'] = proj_cfg._config['LAMMPS_SETUP']
        self._config['MPI'] = proj_cfg._config['MPI']
        self._config['SCHEDULER'] = proj_cfg._config['SCHEDULER']

    def _workflow_wizard(self):
        """
        A procedure to guide the user through setting up a workflow.

        """

        workflows = search_workflows.get_implemented_workflows()

        click.echo('What type of workflow do you want to run?')
        click.echo(self._dashes)
        search_workflows.print_implemented_workflows(workflows=workflows)
        click.echo(self._dashes)
        
        wf_index = camus_log.ask_4_integer_list(workflows)

        workflow = workflows[wf_index - 1].cls

        click.echo(f'Workflow ({wf_index}) selected.\n')
        click.echo(f'Workflow class info:\n')

        space_length = 12
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Module', workflow.__module__, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Class', workflow.__name__, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Tag', workflow.workflow_tag, space_length=space_length))
        click.echo('{:<{space_length}} {:<{space_length}}'.format('Description', workflow.workflow_description, space_length=space_length))

        click.echo(self._dashes)

        self._config['WORKFLOW']['workflow_tag'] = workflow.workflow_tag

        if workflow._required_wf_options:
            pass

        self._phase_list = workflow._get_phases()

        click.echo(f'This workflow uses {len(self._phase_list)} phases.')
        click.echo(self._dashes)

        self._config['WORKFLOW']['current_phase'] = '0'

#    def _workflow_calculation_wizard(self):
#        """
#        A procedure to guide the user through setting up a workflow.
#
#        """
#
#        self._calculation_type_wizard()
#
#        self._config['WORKFLOW']['calculation_type'] = self._calc_type
#        self._config['WORKFLOW']['calculation_method'] = self._calc_method
#        self._config['WORKFLOW']['calculation_engine'] = self._calc_engine
#
#        self._define_calc_options_messages()
#
#        calc_creator = self._calc_class().creator()
#
#        required_calc_cfg = calc_creator._required_calc_cfg
#
#        for key, values in required_calc_cfg.items():
#
#            for value in values:
#
#                if value in ['calculation_type', 'calculation_method', 'calculation_engine']:
#                    pass
#
#                else:
#
#                    click.echo(self._dashes)
#                    message = self._calc_options_messages[value]
#                    self._config['WORKFLOW'][value] = input(f'{message}\n')
#
#        click.echo(self._dashes)
#        scheduler = camus_log.ask_yes_no('Do you want to run your calculations using a job scheduler? [Y/n]\n')
#
#        if scheduler == 'Y':
#            self._config['WORKFLOW']['schedule'] = 'yes'
#
#        else:
#            self._config['WORKFLOW']['schedule'] = 'no'

    def _config_wizard(self):
        """
        Procedure to guide the user after the initialization of the workflow config file.

        """

        click.echo('Starting workflow configuration...')
        click.echo(self._dashes)

        self._workflow_wizard()

        workflow_env_setup = camus_log.ask_yes_no(f'Do you wish to create a workflow-wide environment configuration now? Type "n" to use the configuration from the active project. [Y/n]\n')

        if workflow_env_setup == 'Y':

            click.echo('Note: the values for the steps you skip will be taken from the active project\'s configuration file.')

            lammps_setup = camus_log.ask_yes_no(f'Do you wish to create a workflow-wide LAMMPS configuration now? Type "n" to use the configuration from the active project. [Y/n]\n')

            if lammps_setup == 'Y':
                self._lammps_setup_wizard()
                click.echo(self._dashes)

            else:
                pass

            mpi_setup = camus_log.ask_yes_no(f'Do you wish to create a workflow-wide MPI configuration now? Type "n" to use the configuration from the active project. [Y/n]\n')

            if mpi_setup == 'Y':
                self._mpi_wizard()
                click.echo(self._dashes)

            else:
                pass

            scheduler_setup = camus_log.ask_yes_no(f'Do you wish to create a workflow-wide scheduler configuration now? Type "n" to use the configuration from the active project. [Y/n]\n')

            if scheduler_setup == 'Y':
                self._scheduler_wizard()
                click.echo(self._dashes)

        else:
            click.echo('Using the configuration from the active project.')

        click.echo(f'Workflow configuration successful!')
        click.echo(f'To edit the active workflow config file, see camus wf config --help')
        click.echo(self._dashes)
