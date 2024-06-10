import click

from camus.utils.environment import Environment

@click.command('env', help='Print basic system information')
def env_cli():
    env = Environment()
    env._print_system_info()
