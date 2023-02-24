""" Definition of the Camus class.

This module defines the central object in the CAMUS algorithm.
It should be able to let several other classes to communicate with each other.
Planned classes: Structures, ML, artN, DFT

"""

from camus.structures import Structures

class Camus:

    def __init__(self, structures=None, *args, **kwargs):
        self.Cstructures = Structures(structures)


