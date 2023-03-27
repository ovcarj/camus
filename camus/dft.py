""" Definition of the DFT class.

This module does fuck all right now.

- create dft input structure
-- specifically VASP input -- POSCAR
-[ab] set dft calculation parameters
-[ab] assemble DFT calculation (POSCAR - from create_dft_input, INCAR - based on the parameters, POTCAR - defaultly what Ivor provided (introduce env variable for default) )
- (run)
-[ab] parse dft output (OUTCAR -> .traj)

-- add env variable for deffault DFT directory for testing purposes

!! we should add tracking of the DFT process and check for convergence, I have a bash script for this if it is of any help !!
"""

import os

from abc import ABC, abstractmethod
from ase.io import read

class DFT(ABC):

    def __init__(self, dft_parameters=None):
        """
        Initializes a new DFT object.

        Parameters:
            dft_parameters: (...)

        """

        if dft_parameters is not None:
            self._dft_parameters = dft_parameters
        else:
            self._dft_parameters = {}

    @abstractmethod
    def set_dft_parameters(self, **kwargs):
        ...

    @abstractmethod
    def create_dft_calculation(self, target_directory=None, path_to_potcar=None):
        ...

    @abstractmethod
    def parse_dft_output(self, target_directory=None):
        ...

class VASP(DFT):

    def __init__(self, dft_parameters=None):

        super().__init__(dft_parameters)

    @property
    def dft_parameters(self):
        return self._dft_parameters

    @dft_parameters.setter
    def dft_parameters(self, new_dft_parameters):
        self._dft_parameters = {} # Probably makes more sense this way...
        self._dft_parameters = new_dft_parameters

    @dft_parameters.deleter
    def dft_parameters(self):
        del self._dft_parameters

    def set_dft_parameters(self, **kwargs):
        """ Method that sets parameters to be written in the 
        INCAR file to self._dft_parameters dictionary.

        """
       
        default_parameters= {
            'SPACING': 0.25,
            'EDIFF': '1E-6',
            'ENCUT': 230,
            'ISMEAR': 0,
            'SIGMA': 0.05,
            'PREC': 'Accurate',
            'ALGO': 'Normal',
            'NELMIN': 6,
            'ISYM': 0,
            'LREAL': 'Auto',
            'LWAVE': '.FALSE.',
            'LCHARGE': '.FALSE.',
            'NCORE': 8,
            'METAGGA': 'R2SCAN',
            'LASPH': '.TRUE.',
            'LMIXTAU': '.TRUE.'
            }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self._dft_parameters
        for key, value in default_parameters.items():
            self._dft_parameters[key] = kwargs.pop(key, value)
    
    @staticmethod
    def write_POSCAR(input_structure, target_directory=None):
    #If you're calling this function you obviously want to make a POSCAR out of your trajectory so it would stand to reason you have an input structure.

    #Fucking specorder: from what I can tell the write_vasp method keeps the order from the input trajecotry file unless you specify differently (then it will chose alphabetical order), but seeing as the files throughout the camus algorithm are repeatedly ordered in out desired 'Br I Cs Pb', zet another ordering doesn't seem necessary. 
 
        # Read input structure (temporary index)
        data = read(input_structure, index=0)

        # Set default target_directory 
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_DFT_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_DFT_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Write the POSCAR file to the target directory
        from ase.io import write
        write(os.path.join(target_directory, 'POSCAR'), data, format='vasp')

    #this should probably go into camus.py
    def create_dft_calculation(self, target_directory=None, path_to_potcar=None):
       
        # Set default target_directory 
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_DFT_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_DFT_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.dft_parameters:
            self.set_dft_parameters()

        # Define the INCAR file content
        incar_content = "#DFT_PARAMETERS\n"
        for key, value in self.dft_parameters.items():
            if value is not None:
                incar_content += f"  {key} = {self._dft_parameters[key]}\n"
        incar_content += "/\n"

        # Write the INCAR file to the target directory
        with open(os.path.join(target_directory, 'INCAR'), 'w') as f:
            f.write(incar_content)

        # The path to POTCAR
        if path_to_potcar is None:
            path_to_potcar = os.environ.get('DFT_POTCAR')

###
    # Will go into camus.py but I don't want to fuck with that just yet
    def create_batch_dft(self, base_directory, input_structures=None, dft_parameters=None, prefix='dft', schedule=True, job_filename='sub.sh'):

        if input_structures is None:
            input_structures = [/something/] # from where am I actually getting the DFT structures?

        # Set dft_parameters
        if dft_parameters is not None:
            self.dft_parameters = dft_parameters
        else:
            self.dft_parameters = {}

        # Create base directory if it does not exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]

        # Write the dft files
        for i, structure in enumerate(input_structures):
            target_directory = os.path.join(base_directory, f'{prefix}_{i}')
            self.create_dft_calculation(target_directory=target_directory)
            write_POSCAR(input_structure=structure, target_directory=target_directory)
            if schedule:
                self.Cscheduler.write_submission_script(target_directory=target_directory, filename=job_filename)

    def run_batch_dft(self, base_directory,prefix='dft', save_traj=True, traj_filename='dft_structures.traj', job_filename='sub.sh')

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

            if job_info['job_status'] == 'FINISHED'

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
                        if '' not in line:
                            self.Cscheduler.job_info[f'{job_id}'] = 'NOT CONVERGED'
                            break # where to put this break to go to the next structure?

                    self.Cstructures.dft_set[structure_index] = structures

                else: 
                    self.Cscheduler.job_info[f'{job_id}'] = 'CALCULATION_FAILED'

        # Save the converged DFT structures if specified
        if save_traj:
            write(traj_filename, structures)

                    
###
