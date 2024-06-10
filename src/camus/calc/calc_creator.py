import abc

import os
import click

import camus.utils.utils as camus_utils
import camus.utils.log as camus_log

from camus.cfg.config_calc import ConfigCalc

class CalcCreator(abc.ABC):
    """
    Base class for writing all input files for a calculation given a calculation config file.

    """

    def __init__(self, calc_config_path=None):
        """
        Get required config options. If the calculation config file exists, read its contents.

        Parameters
        ----------
        calc_config_path : None | str
            Path to the calculation config file

        """
        
        self._define_required_cfg()

        if calc_config_path:
            self._cfg = ConfigCalc(calc_config_path)

#        self._check_required_cfg()

#        self.dir = self._cfg._config['CALCULATION']['calc_directory']
#        self.label = self._cfg._config['CALCULATION']['calc_label']

#        self.log = f'{self.dir}/{self.label}.log'

#        self._schedule = True

    @abc.abstractmethod
    def write_inputs(self):
        """
        Writes all necessary input files for a calculation.

        """
        pass

    @abc.abstractmethod
    def write_submission_script(self):
        """
        Writes the appropriate submission script.

        """
        pass

    @abc.abstractmethod
    def create_calculation(self):
        """
        Writes all input files and a submission script if ``self._schedule == True``.

        """

        if self._schedule:
            self.write_submission_script()

        self.write_inputs()

        self._get_logger()
        self._write_2_log(camus_log.timestamp_message(f'Calculation {self.label} created on'))

    @abc.abstractmethod
    def _define_required_cfg(self):
        """
        Defines a dictionary of the config options that must be provided for the calculation to be created.

        """
        pass

    def _check_required_cfg(self):
        """
        Checks if the required config is provided.

        """
        pass

    def _get_logger(self):
        """
        Gets the logger object from the ``self.log`` path

        """

        log_split = self.log.rpartition('/')
        logdir = log_split[0]
        logname = log_split[-1]

        self._logger = camus_log.init_logger(logdir=logdir, logname=logname)

    def _write_2_log(self, logtext):
        """
        Writes to the log file at the ``self.log`` path.

        Parameters
        ----------
        logtext : str
            Text to write in the log file

        """

        self._logger.info(logtext)
