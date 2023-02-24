""" Definition of the Structures class.

This module defines everything related to handling structures in the CAMUS algorithm. It is expected that the structures
are given as a list of ASE atoms objects.

"""

import random

class Structures:

    def __init__(self, structures=None, training_set=None, validation_set=None, test_set=None):
        """
        Initializes a new Structures object with a list of structures and optional training, validation and test sets.

        Parameters:
            structures (list of ASE Atoms objects): List of structures to be stored in the Structures object.
            training (list of ASE Atoms objects): List of structures to be used as the training set.
                Defaults to an empty list.
            validation (list of ASE Atoms objects): List of structures to be used as the validation set.
                Defaults to an empty list.
            test (list of ASE Atoms objects): List of structures to be used as the test set.
                Defaults to an empty list.
        """

        self._structures = structures or []
        self._training_set = training_set or []
        self._validation_set = validation_set or []
        self.test = test_set or []

    @property
    def structures(self):
        return self._structures

    @structures.setter
    def structures(self, new_structures):
        self._structures = new_structures

    @structures.deleter
    def structures(self):
        del self._structures

    @property
    def training_set(self):
        return self._training_set

    @training_set.setter
    def training_set(self, new_structures):
        self._training = new_structures

    @training_set.deleter
    def training_set(self):
        del self._training_set

    @property
    def validation_set(self):
        return self._validation_set

    @validation_set.setter
    def validation_set(self, new_structures):
        self._validation = new_structures

    @validation_set.deleter
    def validation_set(self):
        del self._validation_set

    @property
    def test_set(self):
        return self._test_set

    @test_set.setter
    def test_set(self, new_structures):
        self._test = new_structures

    @test_set.deleter
    def test_set(self):
        del self._test_set

    def find_unique_structures(self, replace_structures=False):
        """Find a set of unique structures by some criterium (to be defined)

        If replace_structures=True, replace the current structures in the Camus object with the unique ones.
        Otherwise, return a set of unique structures.
        """

        unique_structures = self.structures[0] #for testing purposes

        if replace_structures: self.structures = unique_structures
        else: return unique_structures

    def create_datasets(self, training_percent=0.8, validation_percent=0.1, test_percent=0.1, randomize=True):
        """ Separates the structures into training, validation and test sets.

        If randomize=True, randomizes the ordering of structures.
        """

        if training_percent + validation_percent + test_percent != 1.0:
            raise ValueError("Percentages do not add up to 1.0")

        structures = self.structures.copy()

        if randomize:
            random.shuffle(structures)

        num_structures = len(structures)
        num_train = int(num_structures * training_percent)
        num_val = int(num_structures * validation_percent)
        num_test = num_structures - num_train - num_val

        self.training = structures[:num_train]
        self.validation = structures[num_train:num_train+num_val]
        self.test = structures[num_train+num_val:]
