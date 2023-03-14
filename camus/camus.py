""" Definition of the Camus class.

This module defines the central object in the CAMUS algorithm.
It should be able to let several other classes to communicate with each other.
Planned classes: Structures, ML, Sisyphus, DFT, Scheduler

"""

import os
import importlib
import subprocess
import time
import glob

from camus.structures import Structures
from camus.sisyphus import Sisyphus

from ase import Atoms

scheduler_module = importlib.import_module('camus.scheduler')

class Camus:

    def __init__(self, structures=[], artn_parameters={}, lammps_parameters={}, sisyphus_parameters={},
            scheduler='Slurm'):
        """
        Initializes a new Camus object, whose attributes `self.Cname` are instances of `name` classes.
        The methods in this class should allow an interface between the `name` classes.

        """

        self.Cstructures = Structures(structures=structures)
        self.Csisyphus = Sisyphus(artn_parameters, lammps_parameters, sisyphus_parameters)

        scheduler_class = getattr(scheduler_module, scheduler)
        self.Cscheduler = scheduler_class()

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

    def create_batch_minimization(self, base_directory, specorder, input_structures=None, prefix='minimization', schedule=True, job_filename='sub.sh'):
        """
         Method that creates a number of `input_structures` directories in `base_directory` with the names
         `prefix`_(# of structure) that contains all files necessary to minimize a structure. 
         If `input_structures` is not given, self.Cstructures.structures is used.
 
         Parameters:
             base_directory: directory in which to create the directories for LAMMPS minimizations
             specorder: order of atom types in which to write the LAMMPS data file
             input_structures: list of ASE Atoms object which should be minimized 
             prefix: prefix for the names of the minimization directories 
             schedule: if True, write a submission script to each directory
             job_filename: name of the submission script
 
         """

        # Set default input_structures to self.Cstructures.structures
        if input_structures is None:
             input_structures = self.Cstructures.structures

        # Create base directory if it does not exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]

        # Write the minimization files 
        for i, structure in enumerate(input_structures):
            target_directory = os.path.join(base_directory, f'{prefix}_{i}')
            self.create_lammps_minimization(input_structure=structure, target_directory=target_directory, specorder=specorder)
            if schedule:
                self.Cscheduler.write_submission_script(target_directory=target_directory, filename=job_filename)

    def run_batch_minimization(self, base_directory, save_traj=True, traj_filename='minimized_structures.traj',
            max_runtime=600, max_queuetime=3600, job_filename='sub.sh'):
        """
         Intended to be used in conjuction with create_batch_minimization.
         Method that runs all minimizations in subdirectories of `base_directory` and stores the minimized
         structures in self.Cstructures.minimized_set. 
         If `save_traj` is given, a `traj_filename` ASE trajectory file is saved to `base_directory`.
         It is assumed the subdirectories names end with a structure index {_1, _2, ...} so that the ordering
         of structures created by create_batch_minimization is preserved.
 
         Parameters:
             base_directory: directory in which the subdirectories with minimization files are given
             save_traj: if True, a `traj_filename` ASE trajectory file is saved to `base_directory`.
             max_runtime [seconds]: if a calculation is still running after max_runtime, cancel and disregard it
             max_queuetime [seconds]: if calculations are still queueing after max_queuetime, cancel and disregard it
             job_filename: name of the submission script

         """

        # cd to base_directory
        os.chdir(base_directory)

        # Get a list of all the subdirectories sorted by the structure index
        subdirectories = sorted(glob.glob('*'), key=lambda x: int(x.split('_')[-1]))
        
        # Initialize self.Cstructures.minimized_set with None
        self.Cstructures.minimized_set = [None] * len(subdirectories)

        # cd to the subdirectories, submit jobs and remember the structure_index 
        for subdirectory in subdirectories:

            os.chdir(subdirectory)
            self.Cscheduler.run_submission_script(job_filename=job_filename)
            job_id = self.Cscheduler.job_ids[-1]

            cwd = os.getcwd()
            structure_index = int(cwd.split('_')[-1])

            self.Cscheduler.jobs_info[f'{job_id}']['structure_index'] = structure_index

            os.chdir(base_directory)

        # Check job status 
        while len(self.Cscheduler.job_ids) > 0:

            for job_id in self.Cscheduler.job_ids:

                result = subprocess.check_output(['squeue', '-h', '-j', str(job_id)])

                # Job completed (not running anymore)
                if len(result.strip()) == 0:

                    print(f'Job {job_id} has completed')
                    self.Cscheduler.job_ids.remove(job_id)
                    self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'F'

                # Job still exists
                else:
                    self.Cscheduler.check_job_status(job_id, max_queuetime, max_runtime)

            # Wait for some time before checking status again
            time.sleep(10)
