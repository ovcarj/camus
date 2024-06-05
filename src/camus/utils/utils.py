"""
Various Python-related utility functions.

"""

import os
import glob
import re
import sys
import time

import pickle
import copy

import matplotlib.pyplot as plt

import click

import pathlib

from shutil import make_archive, rmtree
from datetime import datetime
from random import randint

def save_to_pickle(object, path_to_file):
    with open(path_to_file, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path_to_file):
    with open(path_to_file, 'rb') as handle:
        return pickle.load(handle)

def create_index_dict(input_list):
    """
    Takes a list as input and returns a dictionary where the keys are the unique elements found in the input list, 
    and the values are lists containing the indices at which each element appears in the input list.
    """

    index_dict = {}
    
    for index, element in enumerate(input_list):
        if element not in index_dict:
            index_dict[element] = [index]
        else:
            index_dict[element].append(index)
            
    return index_dict

def new_list():
    """
    Creates a new list.
    """

    new_list = list()

    return copy.copy(new_list) 

def new_dict():
    """
    Creates a new dictionary.
    """

    new_dict = dict()

    return copy.copy(new_dict) 

def create_plot(xlabel='x', ylabel='y', sizex=15.0, sizey=15.0, fontsize=15):
    """
    Creates a basic empty plot with some frequent settings.
    """
    fig, ax = plt.subplots(figsize=(sizex, sizey))
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(direction='in', which='both', labelsize='large')
    return fig, ax

def get_current_datetime():
    """
    Get current datetime in the ``%Y%m%d-%H%M%S`` format.

    Returns
    -------
    timestr : str
        Current datetime in ``%Y%m%d-%H%M%S`` format.

    """

    timestr = time.strftime("%Y%m%d-%H%M%S")

    return timestr

def convert_time_string(time_string):
    """
    Convert datetime from a string in ``%Y%m%d-%H%M%S`` format to a ``datetime.datetime`` object.

    Parameters
    ----------
    time_string : str
        Datetime string in ``%Y%m%d-%H%M%S`` format. 

    Returns
    -------
    datetime.datetime
        datetime.datetime object obtained from the converted datetime string

    """

    datetime_object = datetime.strptime(time_string, '%Y%m%d-%H%M%S')

    return datetime_object

def get_runtime(start_time, end_time):
    """
    Calculate runtime between ``end_time`` and ``start_time``.

    Parameters
    ----------
    start_time : datetime.datetime
        Start time
    end_time : datetime.datetime
        End time

    Returns
    -------
    delta : str
        Runtime in ``%H:%M:%S`` format

    """

    delta = end_time - start_time

    return str(delta)

def random_number():
    """
    Generate a random number as a string (used for naming download directories and logs).

    Returns
    -------
    random_no : str
        Random number between 0 and 100

    """

    random_no = str(randint(0, 100))

    return random_no

def write_text_to_file(path_to_file, text_data):
    """
    Write text to a file.

    Parameters
    ----------
    path_to_file : str
        Output file path
    text_data : str
        Text data to be written into a file

    """

    with open(path_to_file, 'w+') as f:
        f.write(text_data)

def archive_directory(output_filename, directory_name):
    """
    Create .zip archive from a directory called ``directory_name``.

    Parameters
    ----------
    output_filename : str
        Name of the output .zip directory
    directory_name : str
        Path to the directory to be archived
    """

    if output_filename.endswith('.zip'):
        output_filename = output_filename[:-4]

    make_archive(base_name=output_filename, format='zip', base_dir=directory_name)

def delete_directory(directory_name):
    """
    Delete directory called ``directory_name``.

    Parameters
    ----------
    directory_name : str
        Path to the directory to be deleted
    """

    if directory_exists(directory_name):

        rmtree(directory_name)
    
        if not directory_exists(directory_name):
            click.echo(f'Deleted {directory_name}')

        else:
            click.echo(f'Could not delete {directory_name}')

    else:
        click.echo(f'Directory {directory_name} does not exist')

def directory_exists(directory_path):
    """
    Check if directory ``directory_path`` exists.

    Parameters
    ----------
    directory_path : str
        Path to the directory whose existence is checked
 
    Returns
    ----------
    exists : bool
        True if directory exists, False if it does not.

    """

    exists = os.path.isdir(directory_path)

    return exists

def file_exists(file_path):
    """
    Check if directory ``filepath`` exists.

    Parameters
    ----------
    file_path : str
        Path to the file whose existence is checked
 
    Returns
    ----------
    exists : bool
        True if file exists, False if it does not.

    """

    exists = os.path.isfile(file_path)

    return exists

def is_directory_empty(directory_path):
    """
    Check if directory ``directory_path`` is empty.

    Parameters
    ----------
    directory_path : str
        Path to the directory that is checked for existence of any content
 
    Returns
    ----------
    empty : bool
        True if directory is empty, False if it is not.

    """

    exists = directory_exists(directory_path)

    if exists:
        empty = not os.listdir(directory_path)

    else:
        click.echo(f'Directory {directory_path} does not exist.')
        empty = True

    return empty

def get_files_in_dir(directory_path):
    """
    Get a list of files in a directory (excluding subdirectories).

    Parameters
    ----------
    directory_path : str
        Path to the directory in which to search for files
    Returns
    ----------
    files : list
        List of filepaths in the searched directory

    """
    
    files = glob.glob(f'{directory_path}/*')
    files = [f for f in files if os.path.isfile(f)]

    return files

def delete_directory_files(directory_path):
    """
    Deletes files in a directory (does not delete subdirectories and their file content).

    Parameters
    ----------
    directory_path : str
        Path to the directory whose files are to be deleted

    """

    files = get_files_in_dir(directory_path)

    if files:
        for f in files:
            try:
                os.remove(f)
            except:
                click.echo(f'Could not delete file {f}.')
    else:
        click.echo('Nothing to delete.')
        return

    files_after = get_files_in_dir(directory_path)

    if not files_after:
        click.echo(f'Files in {directory_path} deleted.')

    else:
        click.echo(f'Could not delete all files in {directory_path}')

def delete_directory_content(directory_path):
    """
    Deletes everything in a directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory whose contents are to be deleted

    """

    contents = glob.glob(f'{directory_path}/*')
    
    if len(contents) == 0:
        click.echo('Nothing to delete.')
        return

    for root, dirs, files in os.walk(directory_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            rmtree(os.path.join(root, d))

    contents_after = glob.glob(f'{directory_path}/*')

    if not contents_after:
        click.echo(f'Contents of {directory_path} deleted.')

    else:
        click.echo(f'Could not delete all files in {directory_path}')

def delete_file(file_path):
    """
    Deletes a file given by the path.

    Parameters
    ----------
    file_path : str
        Path to the file to be deleted

    """

    if not file_exists(file_path):
        click.echo('Nothing to delete.')

    else:
        os.remove(file_path)

    if not file_exists(file_path):
        click.echo(f'{file_path} deleted.')
