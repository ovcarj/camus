"""

Various Python-related utility functions.

"""

import pickle
import copy
import matplotlib.pyplot as plt

# Utility functions to save and load pickle files

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

