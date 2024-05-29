import click

from camus.initialize import initialize_camus

@click.command('init', help='Create ``camus`` config, data directory and #TODO databases.')
def init_cli():
    initialize_camus()
