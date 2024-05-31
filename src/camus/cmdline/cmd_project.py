import click

from camus.db.project import Project
from camus.db.db import DB

@click.command('new', help="""Create a new project.

        Example usage:

        camus project new -l example_label -d "optional description"
        camus project new

        If the -l flag is not given, a project wizard will guide the user through the project creation process.
        """)
@click.option('-l', '--label', default=None, help="New project label")
@click.option('-d', '--description', default=None, help="Optional project description")
def new_project(label, description):
    """
    Create a new project, optionally with a given label and description.

    """

    proj = Project()
    proj.create_new_project(label=label, description=description)

@click.command('list', help="List all projects")
def list_all_projects():
    """
    Lists all projects found in the ``projects`` table in the database.

    """

    db = DB()
    db.list_all_projects()

@click.group()
def project_cli(name='project', help='Create, configure, query projects or switch working projects'):
    """
    CLI for creating, deleting and querying camus projects.
    """
    pass

project_cli.add_command(new_project)
project_cli.add_command(list_all_projects)

def main():
    project_cli()

if __name__ == '__main__':
    main()
