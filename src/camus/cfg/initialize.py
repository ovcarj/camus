import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

from camus.cfg.datadir import Datadir
from camus.cfg.config import Config

def create_config():
    """
    Create the ``camus`` config file.
    """

    cfg = Config()
    cfg.create_config_file()

def create_directories():
    """
    Create the ``camus`` data directories.
    """

    ad = Datadir()
    ad.create_directories()

def initialize_camus():
    """
    Create ``camus`` data directories and #TODO databases.
    """

    start_datetime = camus_utils.get_current_datetime()

    start_report = camus_log.camus_start(start_datetime)
    click.echo(start_report)

    create_config()
    create_directories()
    
    click.echo('camus initialization complete!')

    end_report = camus_log.camus_end(start_datetime)
    click.echo(end_report)

if __name__ == '__main__':

    initialize_camus()
