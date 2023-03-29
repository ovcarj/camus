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
import numpy as np
import pickle
import shutil 

from camus.structures import Structures
from camus.sisyphus import Sisyphus

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.io.lammpsrun import read_lammps_dump

scheduler_module = importlib.import_module('camus.scheduler')
dft_module = importlib.import_module('camus.dft')

class Camus:

    def __init__(self, structures=None, artn_parameters=None, lammps_parameters=None, sisyphus_parameters=None,
            scheduler='Slurm', dft='VASP'):
        """
        Initializes a new Camus object, whose attributes `self.Cname` are instances of `name` classes.
        The methods in this class should allow an interface between the `name` classes.

        """
        
        if structures is not None:
            self._structures = structures
        else:
            self._structures = []
       
        if artn_parameters is not None:
            self._artn_parameters = artn_parameters
        else:
            self._artn_parameters = {}
      
        if lammps_parameters is not None:
            self._lammps_parameters = lammps_parameters
        else:
            self._lammps_parameters = {}
     
        if sisyphus_parameters is not None:
            self._sisyphus_parameters = sisyphus_parameters
        else:
            self._sisyphus_parameters = {}

        self.Cstructures = Structures(structures=structures)
        self.Csisyphus = Sisyphus(artn_parameters, lammps_parameters, sisyphus_parameters)

        scheduler_class = getattr(scheduler_module, scheduler)
        self.Cscheduler = scheduler_class()

        dft_class = getattr(dft_module, dft)
        self.Cdft = dft_class()

        # Initialize self.sisyphus_dictionary which will contain all information on the batch Sisyphus calculations
        self.sisyphus_dictionary = {}

        # Initialize self.Cstructures.transitions for batch Sisyphus calculations
        self.Cstructures.transitions = []

    def create_sisyphus_calculation(self, input_structure=None, target_directory=None, initial_lammps_parameters=None, specorder=None, atom_style='atomic'):
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

        # Set initial_lammps_parameters 
        if initial_lammps_parameters is not None:
            self._initial_lammps_parameters = initial_lammps_parameters
        else:
            self._initial_lammps_parameters = {}
 
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
            self.Csisyphus.set_sisyphus_parameters()
        self.Csisyphus.write_sisyphus_script(target_directory=target_directory)

    def create_batch_sisyphus(self, base_directory, specorder, input_structures=None, prefix='sis',
            transition_minimum=0.1, transition_maximum=1.0, transition_step=0.1, delta_e_maximum=0.1, calcs_per_parameters=1, 
            schedule=True, run_command='bash sisyphus.sh ', job_filename='sub.sh', initial_lammps_parameters=None, atom_style='atomic'):
        """
        Method that creates `calc_per_parameters` * `input_structures` * #of_parameter_combinations directories in `base_directory` with the names
        {prefix}_(# of structure)_(transition energy index)_(delta_e_final index)_(calculation_index) that contains all files necessary to 
        perform a Sisyphus calculation.
        
        `transition_energy` range is {transition_minimum + n*transition_step} for every n for which transition_energy < transition_maximum
        `delta_e_final` range is {transition_energy - (n+1)*transition_step} for every n for which delta_e_final > delta_e_maximum

        For each combination of `transition_energy` and delta_e_final, calcs_per_parameters directories are generated.

        If `input_structures` is not given, self.Cstructures.structures is used.
 
        Parameters:
            base_directory: directory in which to create the directories for Sisyphus calculations
            specorder: order of atom types in which to write the LAMMPS data file
            input_structures: list of ASE Atoms object from which Sisyphus starts
            schedule: if True, write a submission script to each directory
            job_filename: name of the submission script
            atom_style: LAMMPS atom style 
 
         """
        if delta_e_maximum > transition_maximum:
            raise ValueError("delta_e_maximum must be greater than or equal to transition_minimum") 

        # Set default input_structures to self.Cstructures.structures
        if input_structures is None:
             input_structures = self.Cstructures.structures
        
        # Set initial_lammps_parameters 
        if initial_lammps_parameters is not None:
            self._initial_lammps_parameters = initial_lammps_parameters
        else:
            self._initial_lammps_parameters = {}

        # Create base directory if it does not exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]

        # Generate list of transition_energies
        transition_energies = np.arange(transition_minimum, transition_maximum, step=transition_step)

        # Initialize self.Csisyphus.sisyphus_parameters
        if not self.Csisyphus.sisyphus_parameters:
            self.Csisyphus.set_sisyphus_parameters()

        # Create batch Sisyphus calculations
        for structure_index, structure in enumerate(input_structures):
            
            for te_index, transition_energy in enumerate(transition_energies):

                # Generate list of delta_e_final energies
                delta_e_finals = np.arange(delta_e_maximum, transition_energy + transition_step, step=transition_step)

                for de_index, delta_e_final in enumerate(delta_e_finals):
                    
                    # Set Sisyphus parameters
                    self.Csisyphus.set_sisyphus_parameters(dE_initial_threshold=str(transition_energy), dE_final_threshold=str(delta_e_final))

                    for calculation in range(calcs_per_parameters):

                        target_directory = os.path.join(base_directory, f'{prefix}_{structure_index}_{te_index}_{de_index}_{calculation}')
                        
                        # Create Sisyphus calculation
                        self.create_sisyphus_calculation(input_structure=structure, target_directory=target_directory, specorder=specorder, 
                                atom_style=atom_style, initial_lammps_parameters=initial_lammps_parameters)

                        if schedule:
                            self.Cscheduler.set_scheduler_parameters(run_command=run_command)
                            self.Cscheduler.write_submission_script(target_directory=target_directory, filename=job_filename)

    def run_batch_sisyphus(self, base_directory, specorder, prefix='sis', 
            max_runtime=120000, max_queuetime=10800, job_filename='sub.sh'):
        """
        TODO: run_batch_sisyphus and run_batch_minimization could easily be split into 2-3 methods (e.g. batch_run, analyze results, etc...) Keeping it as it is for now because it works for fast calculations, but could be an issue for long ones.

        Intended to be used in conjuction with create_batch_sisyphus.
        Method that runs all Sisyphus runs in subdirectories of `base_directory` and stores the found
        transitions in self.Cstructures.transitions . 
        If `save_transitions` is True, a ASE trajectory files are saved to `base_directory/transitions`.
        It is assumed the subdirectories names are as given in create_batch_sisyphus.
 
        Parameters:
            base_directory: directory in which the subdirectories with minimization files are given
            specorder: names of atom types in the LAMMPS minimization
            prefix: prefix of the subdirectory names
            max_runtime [seconds]: if a calculation is still running after max_runtime, cancel and disregard it
            max_queuetime [seconds]: if calculations are still queueing after max_queuetime, cancel and disregard it
            job_filename: name of the submission script

        """
        # cd to base_directory
        os.chdir(base_directory)

        # Make directories to store transitions
        os.mkdir('minima')
        os.mkdir('saddlepoints')
        os.mkdir('transitions')
        os.mkdir('transitions/passed')
        os.mkdir('transitions/failed')

        minima_directory = os.path.abspath('minima')
        saddlepoints_directory = os.path.abspath('saddlepoints')
        transitions_directory = os.path.abspath('transitions')
        transitions_passed_directory = os.path.abspath('transitions/passed')
        transitions_failed_directory = os.path.abspath('transitions/failed')

        # Get a list of all the subdirectories sorted by the structure index, transition energy index, delta_e_final index, calc#
        subdirectories = sorted(glob.glob(f'{prefix}_*'), key=lambda x: [int(i) for i in x.split('_')[1:]])
        
        # cd to the subdirectories, submit jobs and remember the calculation label 
        for calculation_index, subdirectory in enumerate(subdirectories):

            os.chdir(subdirectory)
            self.Cscheduler.run_submission_script(job_filename=job_filename)
            job_id = self.Cscheduler.job_ids[-1]

            cwd = os.getcwd()
            cwd_name = os.path.basename(cwd)
            _, calculation_label = cwd_name.split('_', 1)

            self.Cscheduler.jobs_info[f'{job_id}']['calculation_index'] = calculation_index
            self.Cscheduler.jobs_info[f'{job_id}']['calculation_label'] = calculation_label

            os.chdir(base_directory)

        # Check job status 
        while len(self.Cscheduler.job_ids) > 0:

            for job_id in self.Cscheduler.job_ids:

                result = subprocess.check_output(['squeue', '-h', '-j', str(job_id)])

                # Job completed (not running anymore)
                if len(result.strip()) == 0:

                    print(f'Job {job_id} has completed')
                    self.Cscheduler.job_ids.remove(job_id)
                    self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'FINISHED'

                # Job still exists
                else:
                    self.Cscheduler.check_job_status(job_id, max_queuetime, max_runtime, result)

            # Wait for some time before checking status again
            time.sleep(10)

        # Check if jobs with job_status == 'FINISHED' exited correctly, read the transitions
        # energies & forces, store the structures in self.Cstructures.minimization_set

        for job_id, job_info in self.Cscheduler.jobs_info.items():

            if job_info['job_status'] == 'FINISHED':

                calculation_index = job_info['calculation_index']
                calculation_label = job_info['calculation_label']

                self.sisyphus_dictionary[f'{calculation_label}'] = {}
 
                self.sisyphus_dictionary[f'{calculation_label}']['activation_e_forward'] = None
                self.sisyphus_dictionary[f'{calculation_label}']['delta_e_final_top'] = None
                self.sisyphus_dictionary[f'{calculation_label}']['delta_e_final_initial'] = None
                self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'] = []
                self.sisyphus_dictionary[f'{calculation_label}']['minima_structures'] = [] 
                self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_structures'] = []
                self.sisyphus_dictionary[f'{calculation_label}']['minima_energies'] = None
                self.sisyphus_dictionary[f'{calculation_label}']['all_energies'] = None
                self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_energies'] = None
                self.sisyphus_dictionary[f'{calculation_label}']['basin_counters'] = None
                self.sisyphus_dictionary[f'{calculation_label}']['directory'] = None

                directory = job_info['directory']
                self.sisyphus_dictionary[f'{calculation_label}']['directory'] = directory

                os.chdir(directory)

                sisyphus_log_file = os.path.join(directory, f'{prefix}_{calculation_label}_SISYPHUS.log')

                # Check if the SISYPHUS.log file was generated, parse results

                if os.path.exists(sisyphus_log_file):
                    with open(sisyphus_log_file) as f:
                        sisyphus_log_lines = f.readlines()
                        last_log_line = sisyphus_log_lines[-1]

                    if('exceeded while climbing up the hill.' in last_log_line):
                        self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'FAILED_CLIMB'
                        self.sisyphus_dictionary[f'{calculation_label}']['status'] = 'FAILED_CLIMB'

                    elif('exceeded while going down the hill.' in last_log_line):
                        self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'FAILED_DESCEND'
                        self.sisyphus_dictionary[f'{calculation_label}']['status'] = 'FAILED_DESCEND'

                    elif('MSG_END' in last_log_line):
                        self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'PASSED'
                        self.sisyphus_dictionary[f'{calculation_label}']['status'] = 'PASSED'

                        last_log_line_split = last_log_line.split()
                        self.sisyphus_dictionary[f'{calculation_label}']['activation_e_forward'] = float(last_log_line_split[3])
                        self.sisyphus_dictionary[f'{calculation_label}']['delta_e_final_top'] = float(last_log_line_split[6])
                        self.sisyphus_dictionary[f'{calculation_label}']['delta_e_final_initial'] = float(last_log_line_split[9])

                    else:
                        self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'CALCULATION_FAILED'
                        self.sisyphus_dictionary[f'{calculation_label}']['status'] = 'CALCULATION_FAILED'

                    minima_files = sorted(glob.glob('*minimum*xyz'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    saddlepoint_files = sorted(glob.glob('*saddlepoint*xyz'), key=lambda x: int(x.split('_')[-1].split('.')[0]))

                    if len(minima_files) > 0:

                        minima_energies_file = glob.glob('*minima_energies*')[0]
                        saddlepoint_energies_file = glob.glob('*saddlepoint_energies*')[0]
                        all_energies_file = glob.glob('*all_energies*')[0]
                        basin_counters_file = glob.glob('*basin_counters*')[0]

                        self.sisyphus_dictionary[f'{calculation_label}']['minima_energies'] = np.frombuffer(np.loadtxt(minima_energies_file))
                        self.sisyphus_dictionary[f'{calculation_label}']['all_energies'] = np.frombuffer(np.loadtxt(all_energies_file))
                        self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_energies'] = np.frombuffer(np.loadtxt(saddlepoint_energies_file))
                        self.sisyphus_dictionary[f'{calculation_label}']['basin_counters'] = np.frombuffer(np.loadtxt(basin_counters_file))
                    
                        for i, minimum_filename in enumerate(minima_files):

                            saddlepoint_filename = saddlepoint_files[i]
                            minimum_energy = self.sisyphus_dictionary[f'{calculation_label}']['minima_energies'][i]
                            saddlepoint_energy = self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_energies'][i]

                            # This will be moved to a method in Structures, for now code repeats for minimum and saddlepoint
                            with open(minimum_filename) as f:
                                lines = f.readlines()

                            cell=np.array(lines[1].strip().split(' ')[1:10], dtype='float').reshape(3, 3, order='F')
                            positions = np.loadtxt(minimum_filename, skiprows=2, usecols=(1, 2, 3))
                            forces = np.loadtxt(minimum_filename, skiprows=2, usecols=(4, 5, 6))
                            
                            atom_types = []

                            for line in lines[2:]:
                                atom_id = int(line.strip()[0])
                                atom_types.insert(-1, specorder[atom_id - 1])

                            minimum_atoms=Atoms(symbols=atom_types, positions=positions, pbc=True, cell=cell)
                            minimum_atoms.calc = SinglePointCalculator(minimum_atoms, energy=minimum_energy, forces=forces)

                            with open(saddlepoint_filename) as f:
                                lines = f.readlines()

                            cell=np.array(lines[1].strip().split(' ')[1:10], dtype='float').reshape(3, 3, order='F')
                            positions = np.loadtxt(saddlepoint_filename, skiprows=2, usecols=(1, 2, 3))
                            forces = np.loadtxt(saddlepoint_filename, skiprows=2, usecols=(4, 5, 6))
                            
                            atom_types = []

                            for line in lines[2:]:
                                atom_id = int(line.strip()[0])
                                atom_types.insert(-1, specorder[atom_id - 1])

                            saddlepoint_atoms=Atoms(symbols=atom_types, positions=positions, pbc=True, cell=cell)
                            saddlepoint_atoms.calc = SinglePointCalculator(saddlepoint_atoms, energy=saddlepoint_energy, forces=forces)

                            self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'].append(minimum_atoms)
                            self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'].append(saddlepoint_atoms)
                            self.sisyphus_dictionary[f'{calculation_label}']['minima_structures'].append(minimum_atoms)
                            self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_structures'].append(saddlepoint_atoms)

                        # Write transitions
                        if self.sisyphus_dictionary[f'{calculation_label}']['status'] == 'PASSED':
                            write(os.path.join(transitions_passed_directory, f'{calculation_label}_t.traj'), self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'])
                        elif 'FAILED_' in self.sisyphus_dictionary[f'{calculation_label}']['status']:
                            write(os.path.join(transitions_failed_directory, f'{calculation_label}_t.traj'), self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'])

                        write(os.path.join(minima_directory, f'{calculation_label}_m.traj'), self.sisyphus_dictionary[f'{calculation_label}']['minima_structures'])
                        write(os.path.join(saddlepoints_directory, f'{calculation_label}_s.traj'), self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_structures'])
                    
                    else:
                        self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'NO_STRUCTURES_WRITTEN'
                        self.sisyphus_dictionary[f'{calculation_label}']['status'] = 'NO_STRUCTURES_WRITTEN'

                else:
                    self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'CALCULATION_FAILED'
                    self.sisyphus_dictionary[f'{calculation_label}']['status'] = 'CALCULATION_FAILED'

        with open(os.path.join(f'{base_directory}', 'sisyphus_dictionary.pkl'), 'wb+') as f:
            pickle.dump(self.sisyphus_dictionary, f)

    def create_lammps_minimization(self, input_structure=None, target_directory=None, specorder=None, atom_style='atomic'):
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

    def create_batch_minimization(self, base_directory, specorder, input_structures=None, prefix='minimization', schedule=True, job_filename='sub.sh', atom_style='atomic'):
        """
        Method that creates a number of `input_structures` directories in `base_directory` with the names
        `prefix`_(# of structure) that contains all files necessary to minimize a structure (with LAMMPS). 
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
            self.create_lammps_minimization(input_structure=structure, target_directory=target_directory, specorder=specorder, atom_style=atom_style)
            if schedule:
                self.Cscheduler.write_submission_script(target_directory=target_directory, filename=job_filename)

    def run_batch_minimization(self, base_directory, specorder, prefix='minimization', save_traj=True, 
            traj_filename='minimized_structures.traj', max_runtime=1800, max_queuetime=3600, job_filename='sub.sh'):
        """
         Intended to be used in conjuction with create_batch_minimization.
         Method that runs all minimizations in subdirectories of `base_directory` and stores the minimized
         structures in self.Cstructures.minimized_set. 
         If `save_traj` is True, a `traj_filename` ASE trajectory file is saved to `base_directory`.
         It is assumed the subdirectories names end with a structure index {_1, _2, ...} so that the ordering
         of structures created by create_batch_minimization is preserved.
 
         Parameters:
             base_directory: directory in which the subdirectories with minimization files are given
             specorder: names of atom types in the LAMMPS minimization
             prefix: prefix of the subdirectory names
             save_traj: if True, a `traj_filename` ASE trajectory file is saved to `base_directory`.
             max_runtime [seconds]: if a calculation is still running after max_runtime, cancel and disregard it
             max_queuetime [seconds]: if calculations are still queueing after max_queuetime, cancel and disregard it
             job_filename: name of the submission script

         """

        # cd to base_directory
        os.chdir(base_directory)

        # Get a list of all the subdirectories sorted by the structure index
        subdirectories = sorted(glob.glob(f'{prefix}*'), key=lambda x: int(x.split('_')[-1]))
        
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
                self.Cscheduler.check_job_status(job_id, max_queuetime, max_runtime)

            # Wait for some time before checking status again
            time.sleep(10)

        # Check if jobs with job_status == 'FINISHED' exited correctly, read the minimized structure
        # energies & forces, store the structures in self.Cstructures.minimization_set

        for job_id, job_info in self.Cscheduler.jobs_info.items():

            if job_info['job_status'] == 'FINISHED':

                directory = job_info['directory']
                structure_index = job_info['structure_index']
                minimization_file = os.path.join(directory, 'minimized.xyz')
                log_lammps = os.path.join(directory, 'log.lammps')

                # Check if the minimized.xyz file was generated

                if os.path.exists(minimization_file):
                    with open(minimization_file) as f:
                        structure = read_lammps_dump(f, specorder=specorder)

                    # Get potential energy
                    with open(log_lammps) as f:
                        log_lines = f.readlines()
                    
                    for i, line in enumerate(log_lines):
                        if 'Energy initial, next-to-last, final =' in line:
                            energies_line = log_lines[i+1].strip()
                            break
                    
                    potential_energy = energies_line.split()[-1]
                    structure.calc.results['energy'] = float(potential_energy)
                    self.Cstructures.minimized_set[structure_index] = structure

                else:
                    self.Cscheduler.jobs_info[f'{job_id}'] = 'CALCULATION_FAILED'

        # Save a trajectory file with minimized structures if specified
        if save_traj:
            # Write only succesfully minimized structures
            clean_minimizations = [structure for structure in self.Cstructures.minimized_set if structure is not None]
            write(traj_filename, clean_minimizations)

###
    def create_dft_calculation(self, target_directory=None, path_to_potcar=None):
        """
        Method which in one directory assembles files necessary for a DFT (VASP) calculation (save for POSCAR which is created with Cdft.write_POSCAR)
        If `target_directory` is not given, `$CAMUS_DFT_DIR` is used.
        If `path_to_potcar` is not provided, default POTCAR at `(...)` is used (with specorder: Br I Cs Pb)

        """
       
        # Set default target_directory 
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_DFT_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_DFT_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.Cdft.dft_parameters:
            self.Cdft.set_dft_parameters()

        # Define the INCAR file content
        incar_content = "#DFT_PARAMETERS\n"
        for key, value in self.Cdft.dft_parameters.items():
            if value is not None:
                incar_content += f"  {key} = {self.Cdft._dft_parameters[key]}\n"
        incar_content += "/\n"

        # Write the INCAR file to the target directory
        with open(os.path.join(target_directory, 'INCAR'), 'w') as f:
            f.write(incar_content)

        # The path to POTCAR
        #if path_to_potcar is None:
        #    path_to_potcar = os.environ.get('DFT_POTCAR')

        # Path to POTCAR (temp)
        path_to_potcar = '/home/radovan/Downloads/POTCAR'

        # Copy POTCAR into target_directory
        shutil.copy(path_to_potcar, target_directory)

###
    def create_batch_dft(self, base_directory, input_structures=None, dft_parameters=None, prefix='dft', schedule=True, job_filename='sub.sh'):
        """
        Method that creates a number of `input_structures` directories in `base_directory` with the names
        `prefix`_(# of structure) that contains all files necessary SCF DFT calculation. 
        If `input_structures` is not given, self.Cstructures.structures is used.
 
        Parameters:
            base_directory: directory in which to create the directories for DFT SCF calculation
            input_structures: list of ASE Atoms object which will have an SCF calculation done (intended to be the output of LAMMPs minimization) 
            dft_parameters: parameters which go into an INCAR for a succesful and fast SCF calculation
            prefix: prefix for the names of the minimization directories 
            schedule: if True, write a submission script to each directory
            job_filename: name of the submission script
 
        """

        # Set default input_structues if not specified
        if input_structures is None:
            input_structures = self.Cstructures.structures 

        # Set dft_parameters
        if dft_parameters is not None:
            self.Cdft.dft_parameters = dft_parameters
        else:
            self.Cdft.dft_parameters = {}

        # Create base directory if it does not exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]

        # Write the dft files
        for i, structure in enumerate(input_structures):
            target_directory = os.path.join(base_directory, f'{prefix}_{i}')
            self.create_dft_calculation(target_directory=target_directory)
            self.Cdft.write_POSCAR(input_structure=structure, target_directory=target_directory)
            if schedule:
                #self.Cscheduler.set_scheduler_parameters(run_command=run_command)
                #self.Cscheduler.write_submission_script(target_directory=target_directory, filename=job_filename)
                self.Cdft.write_VASP_sub(target_directory=target_directory, job_filename=job_filename)

    def run_batch_dft(self, base_directory, prefix='dft', save_traj=True, traj_filename='dft_structures.traj', job_filename='sub.sh'):
        """
         Intended to be used in conjuction with create_batch_dft.
         Method that runs all SCF calculations in subdirectories of `base_directory` and stores the converged structures in self.Cstructures.dft_set. 
         If `save_traj` is True, a `traj_filename` ASE trajectory file is saved to `base_directory`.
         It is assumed the subdirectories names end with a structure index {_1, _2, ...} so that the ordering of structures created by create_batch_dft is preserved.
 
         Parameters:
             base_directory: directory in which the subdirectories with minimization files are given
             prefix: prefix of the subdirectory names
             save_traj: if True, a `traj_filename` ASE trajectory file is saved to `base_directory`.
             traj_filename: specifies the name of the output trajectory file
             job_filename: name of the submission script

         """


        # cd to base_directory
        os.chdir(base_directory)

        # Get a list of all the subdirectories sorted by the structure index
        subdirectories = sorted(glob.glob(f'{prefix}*'), key=lambda x: int(x.split('_')[-1]))

        # Initialize self.Cstructures.dft_set with None
        self.Cstructures.dft_set = [None] * len(subdirectories)

        # cd to the subdirectories, submit jobs and rememeber the structure_index
        for subdirectory in subdirectories:

            os.chdir(subdirectory)
            self.Cscheduler.run_submission_script(job_filename=job_filename)
            job_id = self.Cscheduler.job_ids[-1]

            cwd = os.getcwd()
            structure_index - int(cwd.split('_')[-1])

            self.Cscheduler.job_info[f'{job_id}']['structure_index'] = structure_index

            os.chdir(base_directory)

        # Check job status
        while len(self.Cscheduler.job_ids) > 0:

            for job_id in self.Cscheduler.job_ids:

                result = subprocess.check_output(['squeue', '-h', '-j', str(job_id)])

                # Job not running anymore
                if len(result.strip()) == 0:

                    print(f'Job {job_id} has completed.')
                    self.Cscheduler.job_ids.remove(job_id)
                    self.Cscheduler.job_info[f'{job_id}']['job_status'] = 'FINISHED'

                # Job still exists
                else: 
                    self.Cscheduler.check_job_status(job_id, result)

            # Wait a second before checking again
            time.sleep(10)

            # Check if 'FINISHED' jobs exited correctly
            # store the structure along with the calculated energy and forces in self.Cstructures.dft_set

            for job_id, job_info in self.Cscheduler.jobs_info.items():

                if job_info['job_status'] == 'FINISHED':

                    directory = job_info['directory']
                    structure_index = job_info['structure_index']
                    outcar_file = os.path.join(directory, 'OUTCAR')

                    # Check if OUTCAR exists

                    if os.path.exists(outcar_file):
                        with open(outcar_file) as f:
                            structure = read(f)
                            out_lines = f.readlines()

                        # Check for convergence

                        for line in out_lines:
                            if 'Voluntary' in line:
                                self.Cstructures.dft_set[structure_index] = structures
                            # if not there the calculation died somewhere along the way and is thus 'NOT CONVERGED'
                            else:
                                self.Cscheduler.job_info[f'{job_id}'] = 'NOT CONVERGED'

                    else: 
                        self.Cscheduler.job_info[f'{job_id}'] = 'CALCULATION_FAILED'

            # Save the converged DFT structures if specified
            if save_traj:
                write(traj_filename, structures)



