""" Definition of the Camus class.

This module defines the central object in the CAMUS algorithm.
It should be able to let several other classes to communicate with each other.
Planned classes: Structures, ML, Sisyphus, DFT, Scheduler

"""

import os
from camus.structures import Structures
from camus.sisyphus import Sisyphus

class Camus:

    def __init__(self, structures=[], artn_parameters={}, lammps_parameters={}, sisyphus_parameters={}):
        """
        Initializes a new Camus object, whose attributes `self.Cname` are instances of `name` classes.
        The methods in this class should allow an interface between the `name` classes.
        """

        self.Cstructures = Structures(structures=structures)
        self.Csisyphus = Sisyphus(artn_parameters, lammps_parameters, sisyphus_parameters)

    def create_sisyphus_calculation(self, input_structure=None, target_directory=None, initial_lammps_parameters={}, specorder=None, atom_style='atomic'):
        """
        Convenience method that writes all necessary files to start a Sisyphus calculation for an `input_structure` to a `target_directory`.
        If `input_structure` is not given, self.Cstructures.structures[0] is used.
        If `target_directory` is not given, `$CAMUS_SISYPHUS_DATA_DIR` is used
        If self.Csisyphus.{artn_parameters, lammps_parameters, sisyphus_parameters} is an empty dictionary, default values are generated. The provided lammps parameters should be the ones for the main lammps.in input file used by ARTn.
        If initial_lammps_parameters is an empty dictionary, a default initial_lammps.in file is generated. 

        Parameters:
            input_structure: ASE Atoms object for which to write the LAMMPS data file
            target_directory: directory in which to create the Sisyphus calculation
            initial_lammps_parameters: dictionary with the contents of the LAMMPS input file for the initial PE calculation
            specorder: order of atom types in which to write the LAMMPS data file
            atom_style: LAMMPS atom style 

        """

        # Set the default input_structure to self.Cstructures.structures[0]
        if input_structure is None:
            input_structure = self.Cstructures.structures[0]

        # Set the default target directory to CAMUS_SISYPHUS_DATA_DIR environment variable
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_SISYPHUS_DATA_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_SISYPHUS_DATA_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Write artn.in file
        if not self.Csisyphus.artn_parameters:
            self.Csisyphus.set_artn_parameters()
        self.Csisyphus.write_artn_in(target_directory=target_directory)

        # Write initial_lammps.in file for PE calculation
        initial_sisyphus_ins = Sisyphus(lammps_parameters=initial_lammps_parameters)
        if not initial_sisyphus_ins.lammps_parameters:
            initial_sisyphus_ins.set_lammps_parameters(initial_sisyphus=True)
        initial_sisyphus_ins.write_lammps_in(target_directory=target_directory, filename='initial_lammps.in')

        # Write the main lammps.in file used by ARTn
        if not self.Csisyphus.lammps_parameters:
            self.Csisyphus.set_lammps_parameters()
        self.Csisyphus.write_lammps_in(target_directory)

        # Write the lammps.data file
        self.Cstructures.write_lammps_data(target_directory=target_directory, input_structures=input_structure, prefixes='', specorder=specorder, write_masses=True, atom_style=atom_style)

        # Write the Sisyphus bash script 
        if not self.Csisyphus.sisyphus_parameters:
            self.Csisyphus.set_sisyphus_parameters
        self.Csisyphus.write_sisyphus_script(target_directory=target_directory)

    def create_lammps_minimization(self, input_structure=None, target_directory=None, lammps_parameters={}, specorder=None, atom_style='atomic'):
        """
        Convenience method that writes all necessary files to minimize an `input_structure` with LAMMPS. 
        If `input_structure` is not given, self.Cstructures.structures[0] is used.
        If `target_directory` is not given, `$CAMUS_LAMMPS_MINIMIZATION_DIR` is used.
        If self.Csisyphus.lammps_parameters is an empty dictionary, default values for a LAMMPS minimization are generated.

        Parameters:
            input_structures: List of ASE Atoms object for which to write the LAMMPS data file
            target_directory: directory in which to create the files for a LAMMPS minimization
            specorder: order of atom types in which to write the LAMMPS data file
            atom_style: LAMMPS atom style 

        """

        # Set the default input_structure to self.Cstructures.structures[0]
        if input_structure is None:
            input_structure = self.Cstructures.structures[0]

        # Set the default target directory to LAMMPS_MINIMIZATION_DIR environment variable
        if target_directory is None:
            target_directory = os.environ.get('LAMMPS_MINIMIZATION_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and LAMMPS_MINIMIZATION_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Write the lammps.in file for minimization
        if not self.Csisyphus.lammps_parameters:
            self.Csisyphus.set_lammps_parameters(minimization=True)
        self.Csisyphus.write_lammps_in(target_directory)

        # Write the lammps.data file
        self.Cstructures.write_lammps_data(target_directory=target_directory, input_structures=input_structure, prefixes='', specorder=specorder, write_masses=True, atom_style=atom_style)
