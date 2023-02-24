""" Definition of the Structures class.

This module defines everything related to handling structures in the CAMUS algorithm.

"""

class Structures:

    def __init__(self, structures=None):
        self.structures = structures or []

    def find_unique_structures(self, replace_structures=False):
        """Find a set of unique structures by some criterium (to be defined)

        If replace_structures=True, replace the current structures in the Camus object with the unique ones.
        Otherwise, return a set of unique structures.
        """
        unique_structures = self.structures[0] #for testing purposes

        if replace_structures: self.structures = unique_structures
        else: return unique_structures


