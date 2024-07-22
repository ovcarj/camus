import click

from camus.db.workflow_manager import WorkflowManager

@click.command('new', help="""Create a new workflow in the active project

        Example usage:

        camus wf new -l example_label -d "Optional description"

        camus wf new -l abc -f workflowconfig.cfg

        If the -f flag is not given, the user will be guided through the workflow configuration process.
        """)
@click.option('-l', '--label', default=None, help="New workflow label")
@click.option('-d', '--description', default=None, help="Optional workflow description")
@click.option('-f', '--cfgfile', default=None, help="Optional path to a workflow config file")
def new_workflow(label, description, cfgfile):
    """
    Create a new workflow, with a given label and optional description and config file.

    """

    workflow = WorkflowManager()
    workflow.create_new_workflow(label=label, description=description, config_file=cfgfile)

@click.command('list', help="""List all workflows in the active project

        Example usage:

        camus wf list
        
        """)
def list_all_workflows():
    """
    Lists all workflows in the active project found in the ``workflows`` table in the database.

    """

    workflow = WorkflowManager()
    workflow.list_all_workflows()

@click.command('switch', help="""Change the active workflow using a given workflow label

        Example usage: 
        
        camus wf switch my_workflow_label

        To see the list of existing workflows, use: 
        
        camus wf list
        """)
@click.argument('workflow_label')
def switch_active_workflow(workflow_label):
    """
    Changes the active workflow using a given workflow label.

    """

    workflow = WorkflowManager()
    workflow.switch_active_workflow(label=workflow_label)

@click.command('status', help="""Print info on currently active workflow

        Example usage:

        camus wf status

        """)
def print_active_workflow():
    """
    Prints the active workflow.

    """

    workflow = WorkflowManager()
    workflow._print_active_workflow()

@click.command('log', help="""Print the log of the currently active workflow

        Example usage:

        camus wf log

        """)
def print_active_log():
    """
    Prints the log of the active workflow.

    """

    workflow = WorkflowManager(load_active=True)
    workflow._print_log()

@click.command('print', help="""Print the config of the currently active workflow

        Example usage:

        camus wf config print

        """)
def print_active_config():
    """
    Prints the config of the active workflow.

    """

    workflow = WorkflowManager()
    workflow._print_active_config()

@click.command('edit', help="""Edit the the active workflow config file

        Example usage: 
        
        camus wf config edit lammps_exe /path/to/lmp

        The argument after "edit" should be one of the options in the config file.

        The new value should be given after the option.

        NOTE: if you want to pass a string which includes a hyphen as a new value,
        you should prepend the string with "--", e.g.:

        camus wf config edit lammps_run_command -- mpirun -np 2

        To see the current config file, use:

        camus wf config print

        """)
@click.argument('option')
@click.argument('value', nargs=-1)
def edit_config(option, value):
    """
    Edit the active workflow config file by passing option and new value.

    """

    value_str = ' '.join(value)

    workflow = WorkflowManager()
    workflow._edit_active_config_by_subsection(subsection=option, value=value_str)

@click.group()
def config(name='config', help="""Print/edit the config file of the currently active workflow."""):
        """
        Print/edit the config file of the currently active workflow

        Example usage:

        camus wf config print

        camus wf config edit lammps_exe /path/to/lmp/exe
        """
        pass

@click.group()
def workflow_cli(name='wf', help='Create, configure, query workflows or switch working workflows'):
    """
    Create, configure and query workflows in the active project

    """
    pass

config.add_command(print_active_config)
config.add_command(edit_config)

workflow_cli.add_command(new_workflow)
workflow_cli.add_command(list_all_workflows)
workflow_cli.add_command(switch_active_workflow)
workflow_cli.add_command(print_active_workflow)
workflow_cli.add_command(print_active_log)
workflow_cli.add_command(config)

def main():
    workflow_cli()

if __name__ == '__main__':
    main()
