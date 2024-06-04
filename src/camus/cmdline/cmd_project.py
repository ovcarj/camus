import click

from camus.db.project import Project

@click.command('new', help="""Create a new project

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

@click.command('list', help="""List all projects

        Example usage:

        camus project list
        
        """)

def list_all_projects():
    """
    Lists all projects found in the ``projects`` table in the database.

    """

    proj = Project()
    proj.list_all_projects()

@click.command('switch', help="""Change the active project using a given project label

        Example usage: 
        
        camus project switch my_project_label

        To see the list of existing projects, use: 
        
        camus project list
        """)
@click.argument('project_label')
def switch_active_project(project_label):
    """
    Changes the active project using a given project label.

    """

    proj = Project()
    proj.switch_active_project(label=project_label)

@click.command('active', help="""Print which project is currently active
        
        Example usage:

        camus project active

        """)
def print_active_project():
    """
    Prints the active project.
    """

    proj = Project()
    proj.print_active_project()

@click.group()
def project_cli(name='project', help='Create, configure, query projects or switch working projects'):
    """
    CLI for creating, deleting and querying camus projects.
    """
    pass

project_cli.add_command(new_project)
project_cli.add_command(list_all_projects)
project_cli.add_command(switch_active_project)
project_cli.add_command(print_active_project)

def main():
    project_cli()

if __name__ == '__main__':
    main()
