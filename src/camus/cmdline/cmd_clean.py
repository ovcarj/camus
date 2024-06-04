import click

from camus.db.datadir import Datadir
from camus.db.db import DB
from camus.db.project import Project
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
    
    project = Project()
    project.delete_project(label=project_label)

@click.group()
def clean_cli(name='clean', help='Tools for cleaning the camus data directory'):
    """
    Tools for cleaning the camus data directory
    """
    pass

clean_cli.add_command(clean_all_data)
clean_cli.add_command(clean_config)
clean_cli.add_command(clean_db)
clean_cli.add_command(clean_project)

def main():
    clean_cli()

if __name__ == '__main__':
    main()
