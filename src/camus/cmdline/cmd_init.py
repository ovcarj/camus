import click

from camus.initialize import initialize_camus

@click.command('init', help='Create ``camus`` config file, data directory and database.')
def init_cli():
    initialize_camus()
