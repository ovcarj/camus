import click

from camus.db.batch import Batch

@click.command('new', help="""Create a new batch in the active project

        Example usage:

        camus batch new -l example_label -d "Optional description"

        camus batch new -l abc -f batchconfig.cfg

        If the -f flag is not given, the user will be guided through the batch configuration process.
        """)
@click.option('-l', '--label', default=None, help="New batch label")
@click.option('-d', '--description', default=None, help="Optional batch description")
@click.option('-f', '--cfgfile', default=None, help="Optional path to a batch config file")
def new_batch(label, description, cfgfile):
    """
    Create a new batch, with a given label and optional description and config file.

    """

    batch = Batch()
    batch.create_new_batch(label=label, description=description, config_file=cfgfile)

@click.command('list', help="""List all batches in the active project

        Example usage:

        camus batch list
        
        """)
def list_all_batches():
    """
    Lists all batches in the active project found in the ``batches`` table in the database.

    """

    batch = Batch()
    batch.list_all_batches()

@click.command('switch', help="""Change the active batch using a given batch label

        Example usage: 
        
        camus batch switch my_batch_label

        To see the list of existing batches, use: 
        
        camus batch list
        """)
@click.argument('batch_label')
def switch_active_batch(batch_label):
    """
    Changes the active batch using a given batch label.

    """

    batch = Batch()
    batch.switch_active_batch(label=batch_label)

@click.command('show', help="""Print info on currently active batch

        Example usage:

        camus batch show

        """)
def print_active_batch():
    """
    Prints the active batch.

    """

    batch = Batch()
    batch._print_active_batch()

@click.command('log', help="""Print the log of the currently active batch

        Example usage:

        camus batch log

        """)
def print_active_log():
    """
    Prints the log of the active batch.

    """

    batch = Batch(load_active=True)
    batch._print_log()

@click.command('show', help="""Print the config of the currently active batch

        Example usage:

        camus batch config show

        """)
def print_active_config():
    """
    Prints the config of the active batch.

    """

    batch = Batch()
    batch._print_active_config()

@click.command('edit', help="""Edit the the active batch config file

        Example usage: 
        
        camus batch config edit lammps_exe /path/to/lmp

        The argument after "edit" should be one of the options in the config file.

        The new value should be given after the option.

        NOTE: if you want to pass a string which includes a hyphen as a new value,
        you should prepend the value with "--", e.g.:

        camus batch config edit lammps_run_command -- mpirun -np 2

        To see the current config file, use:

        camus batch config show

        """)
@click.argument('option')
@click.argument('value', nargs=-1)
def edit_config(option, value):
    """
    Edit the active batch config file by passing option and new value.

    """

    value_str = ' '.join(value)

    batch = Batch()
    batch._edit_active_config_by_subsection(subsection=option, value=value_str)

@click.group()
def config(name='config', help="""Print/edit the config file of the currently active batch."""):
        """
        Print/edit the config file of the currently active batch

        Example usage:

        camus batch config show

        camus batch config edit lammps_exe /path/to/lmp/exe
        """
        pass

@click.group()
def batch_cli(name='batch', help='Create, configure, query batches or switch working batches'):
    """
    Create, configure and query batches in the active project

    """
    pass

config.add_command(print_active_config)
config.add_command(edit_config)

batch_cli.add_command(new_batch)
batch_cli.add_command(list_all_batches)
batch_cli.add_command(switch_active_batch)
batch_cli.add_command(print_active_batch)
batch_cli.add_command(print_active_log)
batch_cli.add_command(config)

def main():
    batch_cli()

if __name__ == '__main__':
    main()
