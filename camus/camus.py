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

from ase import Atoms
from ase.io import write, read

from camus.structures import Structures
from camus.sisyphus import Sisyphus

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
        Writes all necessary files to start a Sisyphus calculation for an `input_structure` to a `target_directory`.
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
            self.initial_lammps_parameters = initial_lammps_parameters
        else:
            self.initial_lammps_parameters = {}
 
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
            self.initial_lammps_parameters = initial_lammps_parameters
        else:
            self.initial_lammps_parameters = {}

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
                    self.Csisyphus.sisyphus_parameters['dE_initial_threshold'] = str(transition_energy)
                    self.Csisyphus.sisyphus_parameters['dE_final_threshold'] = str(delta_e_final)

                    for calculation in range(calcs_per_parameters):

                        target_directory = os.path.join(base_directory, f'{prefix}_{structure_index}_{te_index}_{de_index}_{calculation}')
                        
                        # Create Sisyphus calculation
                        self.create_sisyphus_calculation(input_structure=structure, target_directory=target_directory, specorder=specorder, 
                                atom_style=atom_style, initial_lammps_parameters=initial_lammps_parameters)

                        if schedule:
                            self.Cscheduler.set_scheduler_parameters(run_command=run_command)
                            self.Cscheduler.write_submission_script(target_directory=target_directory, filename=job_filename)

    def run_batch_sisyphus(self, base_directory, specorder, prefix='sis', job_filename='sub.sh'):
        """
        Intended to be used in conjuction with create_batch_sisyphus.
        Method that runs all Sisyphus runs in subdirectories of `base_directory` and stores the found
        transitions in self.Cstructures.transitions . 
        If `save_transitions` is True, a ASE trajectory files are saved to `base_directory/transitions`.
        It is assumed the subdirectories names are as given in create_batch_sisyphus.
 
        Parameters:
            base_directory: directory in which the subdirectories with minimization files are given
            specorder: names of atom types in the LAMMPS minimization
            prefix: prefix of the subdirectory names
            job_filename: name of the submission script

        """
        # Get cwd so we can return to it at the end of the method
        start_cwd = os.getcwd()

        # cd to base_directory
        os.chdir(base_directory)

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

        # Save initial jobs info to pickle files
        self.Cscheduler.save_to_pickle(self.Cscheduler.jobs_info, os.path.join(f'{base_directory}', 'calculation_info.pkl'))

        os.chdir(start_cwd)

    def parse_batch_sisyphus(self, base_directory, jobs_info, specorder, prefix='sis'):

        # Get cwd so we can return to it at the end of the method
        start_cwd = os.getcwd()

        # Make directories to store transitions
        os.chdir(base_directory)

        os.makedirs('minima', exist_ok=True)
        os.makedirs('saddlepoints', exist_ok=True)
        os.makedirs('transitions', exist_ok=True)
        os.makedirs('transitions/passed', exist_ok=True)
        os.makedirs('transitions/failed', exist_ok=True)

        minima_directory = os.path.abspath('minima')
        saddlepoints_directory = os.path.abspath('saddlepoints')
        transitions_directory = os.path.abspath('transitions')
        transitions_passed_directory = os.path.abspath('transitions/passed')
        transitions_failed_directory = os.path.abspath('transitions/failed')

        # Check if jobs with job_status == 'FINISHED' exited correctly, read the transitions
        # energies & forces, store the structures in self.Cstructures.minimization_set
 
        for job_id, job_info in jobs_info.items():

            if job_info['job_status'] == 'FINISHED':

                calculation_index = job_info['calculation_index']
                calculation_label = job_info['calculation_label']

                # Initialize Sisyphus dictionary

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
                            minimum_atoms = self.Cstructures.parse_sisyphus_xyz(filename=minimum_filename, specorder=specorder)
                            self.sisyphus_dictionary[f'{calculation_label}']['minima_structures'].append(minimum_atoms)

                        for i, saddlepoint_filename in enumerate(saddlepoint_files):
                            saddlepoint_atoms = self.Cstructures.parse_sisyphus_xyz(filename=saddlepoint_filename, specorder=specorder)
                            self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_structures'].append(saddlepoint_atoms)

                        # Combine minima and saddlepoints to transitions
                        self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'] = [None] * (len(minima_files) + len(saddlepoint_files))
                        self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'][1::2] = self.sisyphus_dictionary[f'{calculation_label}']['minima_structures']
                        self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'][::2] = self.sisyphus_dictionary[f'{calculation_label}']['saddlepoints_structures']

                        # Initial sisyphus structure is inserted as a first point of the transition
                        initial_sisyphus_structure = self.Cstructures.parse_lammps_dump(specorder=specorder, log_lammps='initial_lammps.out', dump_name='initial_sisyphus_structure.xyz')
                        self.sisyphus_dictionary[f'{calculation_label}']['transition_structures'].insert(0, initial_sisyphus_structure)

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

        self.Cscheduler.save_to_pickle(self.sisyphus_dictionary, os.path.join(f'{base_directory}', 'sisyphus_dictionary.pkl'))
        os.chdir(start_cwd)

    def create_lammps_calculation(self, input_structure=None, target_directory=None, specorder=None, atom_style='atomic'):
        """
        Writes all necessary files to perform a calculation with LAMMPS. lammps.in parameters are read from self.Csisyphus.lammps_parameters.
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

        # Write the lammps.in file 
        if not self.Csisyphus.lammps_parameters:
            self.Csisyphus.set_lammps_parameters(minimization=True)
        self.Csisyphus.write_lammps_in(target_directory)

        # Write the lammps.data file
        self.Cstructures.write_lammps_data(target_directory=target_directory, input_structures=input_structure, prefixes='', specorder=specorder, write_masses=True, atom_style=atom_style)

    def create_vasp_calculation(self, specorder, input_structure=None, target_directory=None, path_to_potcar=None):
        """
        Writes all files necessary for a DFT (VASP) calculation (save for POSCAR which is created with Cdft.write_POSCAR)
        If `target_directory` is not given, `$CAMUS_DFT_DIR` is used.
        If `path_to_potcar` is not provided, default POTCAR at `(...)` is used. 

        """

        # Set the default input_structure to self.Cstructures.structures[0]
        if input_structure is None:
            input_structure = self.Cstructures.structures[0]
      
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
        
        # Write POSCAR file to target directory
        self.Cdft.write_POSCAR(input_structure=input_structure, target_directory=target_directory, specorder=specorder)

        # The path to POTCAR
        if path_to_potcar is None:
            path_to_potcar = os.environ.get('DFT_POTCAR')

        # Copy POTCAR into target_directory
        try:
            shutil.copy(path_to_potcar, target_directory)
        except:
            raise Exception("POTCAR file required by VASP was not found.")

    def create_batch_calculation(self, base_directory, specorder, calculation_type='LAMMPS',
            input_structures=None, prefix='minimization', schedule=True, job_filename='sub.sh', atom_style='atomic', path_to_potcar=None):

        """
        Creates a number of `input_structures` directories in `base_directory` with the names
        `prefix`_(# of structure) that contains all files necessary to perform a calculation of type `calculation_type`.
        If `input_structures` is not given, self.Cstructures.structures is used.
        Parameters for lammps.in are read from self.Csisyphus.lammps_parameters (defaults to a minimization).
 
        Parameters:
            base_directory: directory in which to create the directories for LAMMPS minimizations
            specorder: order of atom types in which to write the LAMMPS data file
            calculation_type: 'LAMMPS' or 'VASP' (only 'LAMMPS' implemented for now)
            input_structures: list of ASE Atoms object on which the calculation will be performed 
            prefix: prefix for the names of the calculation directories 
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

        # Write the calculation files 
        for i, structure in enumerate(input_structures):
            target_directory = os.path.join(base_directory, f'{prefix}_{i}')

            if calculation_type == 'LAMMPS':
                self.create_lammps_calculation(input_structure=structure, target_directory=target_directory, specorder=specorder, atom_style=atom_style)

            elif calculation_type == 'VASP':
                self.create_vasp_calculation(input_structure=structure, target_directory=target_directory, specorder=specorder, path_to_potcar=path_to_potcar)

            else:
                raise Exception(f"Calculation type {calculation_type} not implemented.")

            if schedule:
                self.Cscheduler.write_submission_script(target_directory=target_directory, filename=job_filename)

    def run_batch_calculation(self, base_directory=None, prefix='minimization', job_filename='sub.sh'):
        """
         Method that submits all calculations in subdirectories of `base_directory`. For now, batch Sisyphus
         is run via a seperate method. 
         It is assumed the subdirectories names end with a structure index {_1, _2, ...} so that the ordering of structures created by create_batch_*_calculation() is preserved.
 
         Parameters:
             base_directory: directory in which the subdirectories with the calculation files are given
             prefix: prefix of the subdirectory names
             job_filename: name of the submission script

         """

        # Assume base_directory = cwd
        if not base_directory:
            base_directory = os.get_cwd()

        # Remember cwd so we can return back at the end of the method
        start_cwd = os.getcwd()

        # cd to base_directory
        os.chdir(base_directory)

        # Get a list of all the subdirectories sorted by the structure index
        subdirectories = sorted(glob.glob(f'{prefix}*/'), key=lambda x: int(os.path.basename(os.path.dirname(x)).split('_')[-1])) 

        # cd to the subdirectories, submit jobs and remember the structure_index 
        for subdirectory in subdirectories:

            os.chdir(subdirectory)
            self.Cscheduler.run_submission_script(job_filename=job_filename)
            job_id = self.Cscheduler.job_ids[-1]

            cwd = os.getcwd()
            structure_index = int(cwd.split('_')[-1])

            self.Cscheduler.jobs_info[f'{job_id}']['structure_index'] = structure_index

            os.chdir(base_directory)

        # Save initial jobs info to pickle files
        self.Cscheduler.save_to_pickle(self.Cscheduler.jobs_info, os.path.join(f'{base_directory}', 'calculation_info.pkl'))

        os.chdir(start_cwd)

    def parse_batch_calculation(self, specorder=None, base_directory=None, jobs_info_dict_filename=None, calculation_type='LAMMPS_minimization',
            save_traj=True, traj_filename='minimized_structures.traj'):
        """
        Note: run self.Cscheduler.check_job_list_status() before parsing to get updated jobs info.
        Parses results from calculations of `calculation_type` in subdirectories of `base_directory`. If `base_directory` is not given,
        cwd is assumed. If `jobs_info_dict_filename` is not given, automatically searches for a file called "*_info.pkl" in `base_directory`.

        If `calculation_type`=='LAMMPS_minimization', calls parse_batch_lammps() and stores structures in self.Cstructures.minimized_set. 
        If `save_traj`=True, a `traj_filename` ASE trajectory file is saved to `base_directory`.
        `specorder` must be given so LAMMPS dump can be read correctly.

        TODO: `calculation_type` == 'LAMMPS_MD',  ...


        """

        # Assume base_directory = cwd
        if not base_directory:
            base_directory = os.get_cwd()

        # Search for jobs_info_dict
        if not jobs_info_dict_filename:
            jobs_info_filename = glob.glob(os.path.join(f'{base_directory}', '*_info.pkl'))[0]

        self.Cscheduler.jobs_info = self.Cscheduler.load_pickle(os.path.join(f'{base_directory}', jobs_info_filename))

        if calculation_type == 'LAMMPS_minimization':

            # Initialize self.Cstructures.minimized_set with None
            self.Cstructures.minimized_set = [None] * len(self.Cscheduler.jobs_info.keys())
            self.parse_batch_lammps(specorder, self.Cscheduler.jobs_info, calculation_type, results_structure_filename='minimized.xyz')

            # Save a trajectory file with minimized structures if specified
            if save_traj:
                # Write only successfully minimized structures
                clean_minimizations = [structure for structure in self.Cstructures.minimized_set if structure is not None]
                write(os.path.join(f'{base_directory}', traj_filename), clean_minimizations)

        elif calculation_type == 'VASP_SCF':
            # Initialize self.Cstructures.dft_set with None
            self.Cstructures.dft_set = [None] * len(self.Cscheduler.jobs_info.keys())
            self.parse_batch_vasp(jobs_info=self.Cscheduler.jobs_info, calculation_type=calculation_type)

            # Save the converged DFT structures if specified
            if save_traj:
                # Write only successful SCFs
                clean_scfs = [structure for structure in self.Cstructures.dft_set if structure is not None]
                write(os.path.join(f'{base_directory}', traj_filename), clean_scfs)

        elif calculation_type == 'Sisyphus':
            self.parse_batch_sisyphus(base_directory=base_directory, jobs_info=self.Cscheduler.jobs_info, specorder=specorder)

        else:
            raise Exception(f"Calculation type {calculation_type} not implemented.")

        self.Cscheduler.save_to_pickle(self.Cscheduler.jobs_info, os.path.join(f'{base_directory}', 'calculation_info.pkl'))

    def parse_batch_lammps(self, specorder, jobs_info, calculation_type, results_structure_filename=None):
        """
        Parses finished LAMMPS calculations in directories given by `jobs_info` dictionary. Only `calculation_type`=='LAMMPS_minimization' is implemented for now.
        """

        for job_id, job_info in jobs_info.items():

            if job_info['job_status'] == 'FINISHED':

                directory = job_info['directory']
                structure_index = job_info['structure_index']

                # LAMMPS minimization case
                if calculation_type == 'LAMMPS_minimization':

                    minimization_file = os.path.join(directory, results_structure_filename)
                    log_lammps = os.path.join(directory, 'log.lammps')

                    # Check if the minimized.xyz file was generated

                    if os.path.exists(minimization_file):
                        self.Cstructures.minimized_set[structure_index] = self.Cstructures.parse_lammps_dump(specorder, log_lammps, minimization_file)

                    else:
                        self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'CALCULATION_FAILED'

                else:
                    raise Exception(f"Calculation type {calculation_type} not implemented.")

    def parse_batch_vasp(self, jobs_info, calculation_type):
        """
        Parses finished VASP calculations in directories given by `jobs_info` dictionary. Only `calculation_type`=='VASP_SCF' is implemented for now.
        """

        for job_id, job_info in jobs_info.items():

            if job_info['job_status'] == 'FINISHED':

                directory = job_info['directory']
                structure_index = job_info['structure_index']

                # VASP SCF case
                if calculation_type == 'VASP_SCF':
                    outcar_file = os.path.join(directory, 'OUTCAR')

                    # Check if OUTCAR exists
                    if os.path.exists(outcar_file):
                        with open(outcar_file) as f:
                            out_lines = f.readlines()

                        # Check for convergence
                        for line in out_lines:
                            if 'Voluntary' in line:
                                self.Cstructures.dft_set[structure_index] = read(outcar_file)

                            else:
                                self.Cscheduler.jobs_info[f'{job_id}'] = 'NOT CONVERGED'

                    else: 
                        self.Cscheduler.jobs_info[f'{job_id}']['job_status'] = 'CALCULATION_FAILED'

                else:
                    raise Exception(f"Calculation type {calculation_type} not implemented.")

