import click

from camus.cfg.config_camus import ConfigCamus

@click.command('print', help='Print the main camus config file.')
def show_config():
    """
    Prints the ``camus`` config file.

    """

    cfg = ConfigCamus()
    cfg.print_config()

@click.command('edit', help="""Edit the contents of the config file.

        Example usage: camus config edit data_directory /home/user/camus_data

        The argument after "edit" should be one of the options in the config file.

        The new value should be given after the option.

        NOTE: if you want to pass a string which includes a hyphen as a new value,
        you should prepend the string with "--", e.g.:

        camus config edit lammps_run_command -- mpirun -np 2

        To see the current config file, use:

        camus config print

        """)
@click.argument('option')
@click.argument('value', nargs=-1)
def edit_config(option, value):
    """
    Edit the ``camus`` config file by passing option and new value

    """

    value_str = ' '.join(value)

    cfg = ConfigCamus()
    cfg.edit_config_by_subsection(subsection=option, value=value_str)

@click.group()
def config_cli(name='config', help='Print/edit the main ``camus`` config file'):
    """
    Print/edit the main ``camus`` config file

    """
    pass

config_cli.add_command(show_config)
config_cli.add_command(edit_config)

def main():
    config_cli()

if __name__ == '__main__':
    main()
