""" Definition of the Camus class.

This module defines the central object in the CAMUS algorithm.

"""

from camus.structures import Structures

class Camus:

    def __init__(self, structures=None, *args, **kwargs):
        self._structures = Structures(structures)

    @property
    def structures(self):
        return self._structures

    @structures.setter
    def structures(self, new_structures):
        self._structures = Structures(new_structures)

    @structures.deleter
    def structures(self):
        del self._structures


