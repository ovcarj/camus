import click

from camus.config import Config

@click.command('show', help='Print the main camus config file.')
def show_config():
    """
    Prints the ``camus`` config file.
    """

    cfg = Config()
    cfg.print_config()

@click.group()
def config_cli(name='config', help='Print and edit the main camus config file'):
    """
    CLI for printing and editing the camus config file
    """
    pass

config_cli.add_command(show_config)

def main():
    config_cli()

if __name__ == '__main__':
    main()
