import click

import platform
import os
import subprocess
import re

import camus.utils.log as camus_log

class Environment:
    """
    Get various system and environment-related information.

    """

    def __init__(self):
        """
        Get basic system and environment information.

        """

        self.system = platform.system()
        self._check_if_unix()
        self._get_distro()

        self._check_camus_tested()

        self._get_paths()

        self._check_has_modules_prog()

        if self._has_modules:
            self._get_loaded_modules()

        self._dashes = camus_log.get_log_dashes()

    def _get_distro(self):
        """
        If ``self.system`` is Linux, get the distribution. 
        Else, set ``self.distro = 'Unknown'``

        """

        if self.system == 'Linux':

            try:
                self.distro = platform.freedesktop_os_release()['NAME']

            except OSError:
                self.distro = 'Unknown'

        else:
            self.distro = 'Unknown'

    def _define_tested_systems(self):
        """
        Define a list of systems on which camus was tested.

        """

        self._tested_systems = ['Linux']

    def _define_tested_distros(self):
        """
        Define a list of Linux distributions on which camus was tested.

        The distribution names are defined as in the ``os-release`` file
        according to the ``freedesktop.org`` standard.

        """

        self._tested_distros = ['CentOS Linux']

    def _check_if_unix(self):
        """
        Checks if the system is UNIX-like.

        """

        # Tested only on Linux for now
        if self.system == 'Linux':
            self._is_unix = True

        else:
            self._is_unix = False

    def _check_camus_tested(self):
        """
        Check if ``camus`` was tested on the current system.

        """

        self._define_tested_systems()
        self._define_tested_distros()

        if self.system in self._tested_systems:

            if self.distro in self._tested_distros:
                self._camus_tested = True
            
            else:
                self._camus_tested = False

        else:
            self._camus_tested = False

    def _get_paths(self):
        """
        Gets the $PATH and $LD_LIBRARY_PATH system variables.

        """

        self.path = os.environ['PATH']

        try:
            self.ld_path = os.environ['LD_LIBRARY_PATH']

        except: 
            self.ld_path = None

    def _check_has_modules_prog(self):
        """
        Checks if the ``module`` environment program exists.

        """

        success_return_code = 0

        subproc_res = subprocess.run(['module --version'], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if subproc_res.returncode == success_return_code:
            self._has_modules = True

        else:
            self._has_modules = False

    def _get_loaded_modules(self):
        """
        Gets the list of loaded modules.

        """

        self.loaded_modules = []

        subproc_res = subprocess.run(['module list'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        output = subproc_res.stdout

        if 'No modules loaded' in output:
            pass

        elif 'Currently Loaded Modules:' in output:

            # Parse only lines of the output that have a r'\d\)' pattern
            modules_concatenated = [x for x in output.split('\n') if re.search(r'\d+\)', x)]

            # Separate each module and get the order in which they are written

            modules_unordered = []
            list_order = []

            for mc in modules_concatenated:

                mc_split = re.split(r'(\d+\))', mc)

                for i, split in enumerate(mc_split):

                    search_pattern = re.search(r'\d+\)', split)

                    if search_pattern:

                        list_order.append(int(search_pattern.group()[:-1]))
                        modules_unordered.append(mc_split[i + 1].strip())
            
            # Order modules

            self.loaded_modules = []

            for order in list_order:
                self.loaded_modules.append(modules_unordered[order - 1])

    def _print_system_info(self):
        """
        Print relevant system info.

        """

        print_length = 15
        
        click.echo('{:<{print_length}} {:<{print_length}}'.format('System', self.system, print_length=print_length))

        if self._is_unix:
            click.echo('{:<{print_length}} {:<{print_length}}'.format('Distribution', self.distro, print_length=print_length))

        click.echo('{:<{print_length}} {:<{print_length}}'.format('camus_tested', str(self._camus_tested), print_length=print_length))

        click.echo('{:<{print_length}} {:<{print_length}}'.format('has_modules', str(self._has_modules), print_length=print_length))

        click.echo(self._dashes)
        click.echo(f'PATH={self.path}')
        click.echo(self._dashes)

        if self.ld_path:

            click.echo(f'LD_LIBRARY_PATH={self.ld_path}')

        else:

            if self._is_unix:
                click.echo('LD_LIBRARY_PATH is not set')

            else:
                pass

        if self._has_modules:

            click.echo(self._dashes)

            if self.loaded_modules:

                click.echo('Currently loaded modules:\n')

                for i, loaded_module in enumerate(self.loaded_modules):
                    click.echo(f'{i}) {loaded_module}')

            else:
                click.echo('No modules are loaded')
