""" Definition of the Camus class.

This module defines the central object in the CAMUS algorithm.

"""

from camus.structures import Structures

class Camus:

    def __init__(self, structures=None, *args, **kwargs):
        self.Cstructures = Structures(structures)


