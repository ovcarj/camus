import click

from camus.db.datadir import Datadir
from camus.db.db import DB
from camus.db.project_manager import ProjectManager
from camus.db.workflow_manager import WorkflowManager
from camus.cfg.config_camus import ConfigCamus

@click.command('all_data', help='Delete the camus data directory.')
def clean_all_data():
    """
    Deletes all ``camus`` data directories
    """

    ad = Datadir()
    ad.clean_all_directories()

@click.command('config', help='Delete the camus config file.')
def clean_config():
    """
    Deletes the ``camus`` config file.
    """

    cfg = ConfigCamus()
    cfg.clean_config()

@click.command('db', help='Delete the camus database.')
def clean_db():
    """
    Deletes the ``camus`` database.
    """

    db = DB()
    db.clean_database()

@click.command('project', help="""Delete the project with the given project label.

        Example usage: camus clean project my_project_label

        To view the list of all projects, use:

        camus project list
        """)
@click.argument('project_label')
def clean_project(project_label):
    """
    Deletes the project with the given ``project_label``.
    """
    
    project = ProjectManager()
    project.delete_project(label=project_label)

@click.command('wf', help="""Delete the workflow with the given workflow label.

        Example usage: camus clean wf my_workflow_label

        To view the list of all workflows in the active project, use:

        camus wf list
        """)
@click.argument('workflow_label')
def clean_workflow(workflow_label):
    """
    Deletes the workflow with the given ``workflow_label``.
    """
    
    workflow = WorkflowManager()
    workflow.delete_workflow(label=workflow_label)

@click.group()
def clean_cli(name='clean', help='Tools for cleaning the ``camus`` data directory'):
    """
    Tools for cleaning the ``camus`` data directory
    """
    pass

clean_cli.add_command(clean_all_data)
clean_cli.add_command(clean_config)
clean_cli.add_command(clean_db)
clean_cli.add_command(clean_project)
clean_cli.add_command(clean_workflow)

def main():
    clean_cli()

if __name__ == '__main__':
    main()
