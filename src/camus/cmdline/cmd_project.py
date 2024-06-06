import click

from camus.db.project import Project

@click.command('new', help="""Create a new project

        Example usage:

        camus project new -l example_label -d "optional description"

        If the -f flag is not given, a project wizard will guide the user through the project configuration process.
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

@click.command('active', help="""Print info on currently active project

        Example usage:

        camus project active

        """)
def print_active_project():
    """
    Prints the active project.

    """

    proj = Project()
    proj._print_active_project()

@click.command('log', help="""Print the log of the currently active project

        Example usage:

        camus project log

        """)
def print_active_log():
    """
    Prints the log of the active project.

    """

    proj = Project(load_active=True)
    proj._print_log()

@click.command('show', help="""Print the config of the currently active project

        Example usage:

        camus project config show

        """)
def print_active_config():
    """
    Prints the config of the active project.

    """

    proj = Project()
    proj._print_active_config()

@click.command('edit', help="""Edit the the active project config file

        Example usage: 
        
        camus project config edit lammps_exe /path/to/lmp

        The argument after "edit" should be one of the options in the config file.

        The new value should be given after the option.

        NOTE: if you want to pass a string which includes a hyphen as a new value,
        you should prepend the value with "--", e.g.:

        camus project config edit lammps_run_command -- mpirun -np 2

        To see the current config file, use:

        camus project config show

        """)
@click.argument('option')
@click.argument('value', nargs=-1)
def edit_config(option, value):
    """
    Edit the active project config file by passing option and new value.

    """

    value_str = ' '.join(value)

    proj = Project()
    proj._edit_active_config_by_subsection(subsection=option, value=value_str)

@click.group()
def config(name='config', help="""Print/edit the config file of the currently active project."""):
        """
        Print/edit the config file of the currently active project.

        Example usage:

        camus project config show

        camus project config edit lammps_exe /path/to/lmp/exe
        """
        pass

@click.group()
def project_cli(name='project', help='Create, configure, query projects or switch working projects'):
    """
    CLI for creating, deleting and querying camus projects.

    """
    pass

config.add_command(print_active_config)
config.add_command(edit_config)

project_cli.add_command(new_project)
project_cli.add_command(list_all_projects)
project_cli.add_command(switch_active_project)
project_cli.add_command(print_active_project)
project_cli.add_command(print_active_log)
project_cli.add_command(config)

def main():
    project_cli()

if __name__ == '__main__':
    main()
