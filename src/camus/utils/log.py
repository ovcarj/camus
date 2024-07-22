import sys
import os
import logging

import click

import camus.utils.utils as camus_utils

from camus import __version__

def init_logger(logname='camus', logdir=None):
    """
    Initialize the logger object.

    Parameters
    ----------
    logname : str
        Name of the logger object (also used in naming in the log file)
    logdir : str
        Directory in which to store the log file. If not given, default to current directory

    Returns
    ----------
    logger : logging.Logger
        logging.Logger object

    """

    if logdir is None:
        logdir = os.getcwd()

    os.makedirs(logdir, exist_ok=True)

    if not logdir.endswith('/'):
        logdir += '/'

    if not logname.endswith('.log'):
        logname += '.log'

    logger = logging.getLogger(logname)

    logger.setLevel(logging.INFO) 

    file_handler = logging.FileHandler(f'{logdir}{logname}')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger

def get_logger_fh_path(logger):
    """
    Get path to where a log is stored 
    (baseFilename of the logger FileHandler with level INFO).

    Parameters
    ----------
    logger : logging.Logger
        logging.Logger object

    Returns
    ----------
    base_filename : str
        Path to where the log is stored.

    """

    if logger.hasHandlers():

        handlers = logger.handlers

        base_filename = ''

        for handler in handlers:

            if type(handler) == logging.FileHandler:
                if handler.level == 20:
                    base_filename += handler.baseFilename

        if base_filename == '':
            return 'No logger file handlers found'

        else:
            return base_filename

    else:
        return 'No logger handlers found.'

def camus_start(start_datetime=None, start_message=None):
    """
    Report on the beginning of ``camus`` execution.

    Parameters
    ----------
    start_datetime : str | None
        Starting datetime in ``%Y%m%d-%H%M%S`` format
    start_message : str | None
        Message to prepend to the start_datetime. If not given, 'camus started on' will be used

    Returns
    ----------
    start_report : str
        Report on the beginning of ``camus`` execution
 
    """

    if not start_datetime:
        start_datetime = camus_utils.get_current_datetime()

    if not start_message:
        start_message = 'camus started on'

    start_report = ''

    log_dashes = get_log_dashes()
    logo = get_logo()

    start_report += logo + '\n'
    start_report += log_dashes + '\n'
    start_report += timestamp_message(message=start_message, datetime=start_datetime)
    start_report += '\n'
    start_report += log_dashes 

    return start_report

def camus_end(start_datetime):
    """
    Report on the end of ``camus`` execution.

    Parameters
    ----------
    start_datetime : str
        Starting datetime in ``%Y%m%d-%H%M%S`` format

    Returns
    ----------
    end_report : str
        Report on the end of ``camus`` execution
 
    """

    end_datetime = camus_utils.get_current_datetime()

    end_report = ''

    log_dashes = get_log_dashes()

    runtime = camus_utils.get_runtime(
            camus_utils.convert_time_string(start_datetime),
            camus_utils.convert_time_string(end_datetime)
            )

    end_report += log_dashes + '\n'
    end_report += f'camus finished on {end_datetime}\n'
    end_report += f'Total runtime: {runtime}'

    return end_report

def timestamp_message(message, datetime=None):
    """
    Return a string of form ``f'{message} {datetime}'``

    Parameters
    ----------
    message : str
        Message to be printed
    datetime : str | None
        Datetime to be printed. If None, use current datetime

    Returns
    -------
    timestamp_message : str
        A string of form f'{message} {datetime}'

    """

    if not datetime:
        datetime = camus_utils.get_current_datetime()

    timestamp_message = f'{message} {datetime}'

    return timestamp_message

def ask_yes_no(question):
    """
    Forces the user to type 'Y' or 'n' and returns the answer.

    Parameters
    ----------
    question : str
        Question to ask the user

    Returns
    -------
    answer : str
        'Y' or 'n'

    """

    dashes = get_log_dashes()

    input_ok = False

    while(input_ok == False):

        answer = input(question)

        click.echo(dashes)

        try:
            assert (answer == 'Y' or answer == 'n')
            input_ok = True

        except AssertionError:
            click.echo('Please enter "Y" or "n".')
            click.echo(dashes)

    return answer

def ask_4_integer_list(lst):
    """
    Given a list ``lst`` of length N, force the user 
    to type an integer in [1, ..., N] and return the answer. 

    Parameters
    ----------
    lst : list
        List of length N

    Returns
    -------
    answer : int 
        Integer in range [1, ..., N]

    """

    dashes = get_log_dashes()

    input_ok = False

    lst_len = len(lst)
    accept_list = [str(i) for i in range(1, lst_len + 1)]

    while(input_ok == False):

        answer = input(f'Please enter an integer in range [1 - {lst_len}].\n')
        click.echo(dashes)

        try:
            assert answer in accept_list
            input_ok = True

        except AssertionError:
            click.echo(f'Invalid integer.')

    return int(answer)

def get_log_dashes():
    """
    Create some dashes for logging.

    Returns
    -------
    str
        Dashes for logging

    """

    return '----------------------------------------------'

def get_short_log_dashes():
    """
    Create some dashes for logging.

    Returns
    -------
    str
        Dashes for logging

    """

    return '-----------------------------'

def get_very_short_log_dashes():
    """
    Create some dashes for logging.

    Returns
    -------
    str
        Dashes for logging

    """

    return '-----------'

def get_logo():
    """
    Create the ``camus`` logo.

    Returns
    -------
    logo : str
        The ``camus`` logo

    """

    logo = rf"""
#
#    ________   __  _____  ______
#   / ___/ _ | /  |/  / / / / __/
#  / /__/ __ |/ /|_/ / /_/ /\ \  
#  \___/_/ |_/_/  /_/\____/___/  
#                                
#                   v={__version__}
#
"""

    return logo
