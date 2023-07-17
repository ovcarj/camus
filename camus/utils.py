import pickle

# Utility functions to save and load pickle files

def save_to_pickle(object, path_to_file):
    with open(path_to_file, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path_to_file):
    with open(path_to_file, 'rb') as handle:
        return pickle.load(handle)
