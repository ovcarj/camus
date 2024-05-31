import click

from camus.cfg.datadir import Datadir
from camus.cfg.config_camus import Config_camus
from camus.db.db import DB

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

    cfg = Config_camus()
    cfg.clean_config()

@click.command('db', help='Delete the camus database.')
def clean_db():
    """
    Deletes the ``camus`` database.
    """

    db = DB()
    db.clean_database()

@click.group()
def clean_cli(name='clean', help='Tools for cleaning the camus data directory'):
    """
    Tools for cleaning the camus data directory
    """
    pass

clean_cli.add_command(clean_all_data)
clean_cli.add_command(clean_config)
clean_cli.add_command(clean_db)

def main():
    clean_cli()

if __name__ == '__main__':
    main()
