""" Definition of the Camus class.

This module defines the central object in the CAMUS algorithm.
It should be able to let several other classes to communicate with each other.
Planned classes: Structures, ML, Sisyphus, DFT

"""

import os
from camus.structures import Structures
from camus.sisyphus import Sisyphus

class Camus:

    def __init__(self, structures=[], *args, **kwargs):
        """
        Initializes a new Camus object, whose attributes `self.Cname` are instances of `name` classes.
        The methods in this class should allow an interface between the `name` classes.
        """

        self.Cstructures = Structures(structures)
        self.Csisyphus = Sisyphus()



