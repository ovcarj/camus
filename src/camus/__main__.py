__version__ = '0.0.1'

import click
from camus.cmdline import cmd_init, cmd_clean, cmd_config, cmd_project, cmd_batch

@click.group(name='camus')
def camus_cli():
    pass

camus_cli.add_command(cmd_init.init_cli, name='init')
camus_cli.add_command(cmd_clean.clean_cli, name='clean')
camus_cli.add_command(cmd_config.config_cli, name='config')
camus_cli.add_command(cmd_project.project_cli, name='project')
camus_cli.add_command(cmd_batch.batch_cli, name='batch')
