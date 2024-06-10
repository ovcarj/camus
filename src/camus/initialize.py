import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log
from camus.utils.environment import Environment

from camus.cfg.config_camus import ConfigCamus
from camus.db.datadir import Datadir
from camus.db.db import DB

def create_config():
    """
    Create the main ``camus`` config file.

    """

    cfg = ConfigCamus()
    cfg.create_config_file()

def create_directories():
    """
    Create the ``camus`` data directories.

    """

    ad = Datadir()
    ad.create_directories()

def create_database():
    """
    Create the ``camus`` database.

    """

    db = DB()
    db.create_database()

def print_env_info():
    """
    Print basic system and environment info.

    """

    env = Environment()
    env._print_system_info()

def initialize_camus():
    """
    Create ``camus`` data directories and database, print environment info.

    """

    start_datetime = camus_utils.get_current_datetime()

    start_report = camus_log.camus_start(start_datetime)
    click.echo(start_report)

    print_env_info()
    click.echo(camus_log.get_log_dashes())

    create_config()
    create_directories()
    create_database()

    click.echo(camus_log.get_log_dashes())
    click.echo('camus initialization complete!')

    end_report = camus_log.camus_end(start_datetime)
    click.echo(end_report)

if __name__ == '__main__':

    initialize_camus()
