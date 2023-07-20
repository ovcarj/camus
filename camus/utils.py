import pickle

# Utility functions to save and load pickle files

def save_to_pickle(object, path_to_file):
    with open(path_to_file, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path_to_file):
    with open(path_to_file, 'rb') as handle:
        return pickle.load(handle)

def create_index_dict(input_list):
    """
    Takes a list as input and returns a dictionary where the keys are the unique elements found in the input list, and the values are lists containing the indices at which each element appears in the input list.

    """

    index_dict = {}
    
    for index, element in enumerate(input_list):
        if element not in index_dict:
            index_dict[element] = [index]
        else:
            index_dict[element].append(index)
            
    return index_dict
