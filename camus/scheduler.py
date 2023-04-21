""" Definition of the Scheduler class.

This module defines everything related to handling cluster scheduling.

"""

import os
import subprocess
import time
import pickle
import glob

from abc import ABC, abstractmethod

class Scheduler(ABC):

    def __init__(self, scheduler_parameters=None):
        """
        Initializes a new Scheduler object.

        Parameters:
            scheduler_parameters: parameters to be written in the sub.sh file.

        """

        if scheduler_parameters is not None:
            self._scheduler_parameters = scheduler_parameters
        else:
            self._scheduler_parameters = {}

        self.job_ids = []
        
        # Meant to be a flexible dictionary, e.g. in the form {'job_id': {working_directory: ..., job_status: ..., ...}}
        self.jobs_info = {} 

    @abstractmethod
    def set_scheduler_parameters(self, **kwargs):
        ...

    @abstractmethod
    def write_submission_script(self, target_directory, filename):
        ...

    @abstractmethod
    def run_submission_script(self, filename):
        ...

    @abstractmethod
    def check_job_status(self, job_id, max_queuetime, max_runtime, *args):
        ...

class Slurm(Scheduler):


    def __init__(self, scheduler_parameters=None):

        super().__init__(scheduler_parameters)

    @property
    def scheduler_parameters(self):
        return self._scheduler_parameters

    @scheduler_parameters.setter
    def scheduler_parameters(self, new_scheduler_parameters):
        self._scheduler_parameters = {} # Probably makes more sense this way...
        self._scheduler_parameters = new_scheduler_parameters

    @scheduler_parameters.deleter
    def scheduler_parameters(self):
        del self._scheduler_parameters

    def set_scheduler_parameters(self, input_file='lammps.in', output_file='lammps.out', **kwargs):
        """ Method that sets parameters to be written in the submission script
        to self._scheduler_parameters dictionary.

        """
        
        # Get common strings from environment variables
        run_lammps = os.environ.get('RUN_LAMMPS')
        lammps_exe = os.environ.get('LAMMPS_EXE')
        lammps_flags = os.environ.get('LAMMPS_FLAGS')

        default_parameters= {
            'partition': 'normal',
            'job_name': 'sisyphus',
            'output': '%x-%j.out',
            'error': '%x-%j.err',
            'mem': '7gb', 
            'nodes': '1',
            'ntasks': '1',
            'modules': ['gnu9', 'openmpi4/4.1.1'],
            'additional_commands': ['export OMP_NUM_THREADS=$SLURM_NTASKS', 'ulimit -s unlimited'],
            'run_command': f'{run_lammps} {lammps_exe} {lammps_flags} -in {input_file} > {output_file}' 
            }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self._scheduler_parameters
        for key, value in default_parameters.items():
            self._scheduler_parameters[key] = kwargs.pop(key, value)


    def write_submission_script(self, target_directory, filename='sub.sh'):
        """ Method that writes a Slurm submission script to `target directory/filename`.
        Parameters:
            target_directory: directory in which to write the submission script
            filename: name of the submission script

        """

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.scheduler_parameters:
            self.set_scheduler_parameters()

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Write the submission script to the target directory
        with open(os.path.join(target_directory, filename), 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --partition={self.scheduler_parameters['partition']}
#SBATCH --job-name={self.scheduler_parameters['job_name']}
#SBATCH --output={self.scheduler_parameters['output']}
#SBATCH --error={self.scheduler_parameters['error']}
#SBATCH --mem={self.scheduler_parameters['mem']}
#SBATCH --nodes={self.scheduler_parameters['nodes']}
#SBATCH --ntasks={self.scheduler_parameters['ntasks']}

module purge
""")

            for module in self.scheduler_parameters['modules']:
                f.write(f"module load {module}\n")

            f.write("\n")

            for command in self.scheduler_parameters['additional_commands']:
                f.write(f"{command}\n")

            f.write(f"\n{self.scheduler_parameters['run_command']}")

    def run_submission_script(self, job_filename='sub.sh'):
        # Submit job
        output = subprocess.check_output(f'sbatch {job_filename}', shell=True)

        # Get job_id, cwd and submission time
        job_id = int(output.split()[-1])
        self.job_ids.append(job_id)
        cwd = os.getcwd()
        submission_time = time.time()

        # Initialize self.jobs_info dictionary

        self.jobs_info[f'{job_id}'] = {'directory': cwd, 'job_status': 'I', 
                'submission_time': submission_time, 'start_time': 0, 'queue_time': 0, 'run_time': -1}

    def check_job_status(self, job_id, max_queuetime, max_runtime):
        """ Checks whether a job with `job_id` is queueing, running or failed.
            If max_queuetime or max_runtime (in seconds) has ellapsed, cancels the job.
        """

        current_time = time.time()

        try:
            squeue_result = subprocess.check_output(['squeue', '-h', '-j', str(job_id)], stderr=subprocess.DEVNULL)
        except:
            squeue_result = b''

        try:
            sjob_result = subprocess.check_output(['sacct', '-j', str(job_id), '--format=state'], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")

        # Job completed (not running anymore)
        if len(squeue_result.strip()) == 0:
            self.job_ids.remove(job_id)
            self.jobs_info[f'{job_id}']['job_status'] = 'FINISHED'

            # If job is 'FINISHED' check if it wasn't explicitly cancelled by a user or administrator
            if self.jobs_info[f'{job_id}']['job_status'] == 'FINISHED':
                job_state = sjob_result.strip().split()[3].decode('utf-8')
                if job_state == 'CANCELLED':
                    self.jobs_info[f'{job_id}']['job_status'] = 'JOB_CANCELLED'
 
        # Job still running
        else:

            self.jobs_info[f'{job_id}']['job_status'] = squeue_result.strip().split()[4].decode('utf-8')

            # Job in queue
            if self.jobs_info[f'{job_id}']['job_status'] == 'PD':

                self.jobs_info[f'{job_id}']['queue_time'] = current_time - self.jobs_info[f'{job_id}']['submission_time']

            # Job waited too long
                if self.jobs_info[f'{job_id}']['queue_time'] > max_queuetime:
                    self.jobs_info[f'{job_id}']['job_status'] = 'MAX_QUEUE_TIME_ELLAPSED'
                    self.job_ids.remove(job_id)
                    subprocess.run(["scancel", job_id])

            # Job failed for some reason
            elif self.jobs_info[f'{job_id}']['job_status'] in ['BF', 'DL', 'CA', 'F', 'NF', 'PR', 'ST', 'TO']:
                self.job_ids.remove(job_id)
                subprocess.run(["scancel", job_id])
            
            # Job running
            elif self.jobs_info[f'{job_id}']['job_status'] == 'R':
                
                # Check if this is the first instance of seeing the job running
                if self.jobs_info[f'{job_id}']['run_time'] == -1:
                    self.jobs_info[f'{job_id}']['start_time'] = current_time
                    self.jobs_info[f'{job_id}']['run_time'] = 0

                self.jobs_info[f'{job_id}']['run_time'] = current_time - self.jobs_info[f'{job_id}']['start_time']

                # Job running too long
                if self.jobs_info[f'{job_id}']['run_time'] > max_runtime:
                    self.jobs_info[f'{job_id}']['job_status'] = 'MAX_RUN_TIME_ELLAPSED'
                    self.job_ids.remove(job_id)
                    subprocess.run(["scancel", job_id])

            # Transient job status - hopefully nothing special is happening
            else: pass

    def check_job_list_status(self, base_directory=None, jobs_info_filename=None, max_runtime=180000, max_queuetime=360000, sleep_time=60):
        """ Calls self.check_job_status every `sleep_time` seconds and checks the status of all jobs in subdirectories of 
            `base_directory`. If `base_directory` is not given, assume `base_directory` = cwd.
            If `jobs_info_filename` is not given, automatically searches for a file named "*_info.pkl" where all the job ids
            and other job info should be written.
            If max_queuetime or max_runtime (in seconds) has ellapsed, cancels the job.
        """

        # Assume base_directory = cwd
        if not base_directory:
            base_directory = os.get_cwd()

        # Search for jobs_info_dict
        if not jobs_info_filename:
            jobs_info_filename = glob.glob(os.path.join(f'{base_directory}', '*_info.pkl'))[0]

        self.jobs_info = self.load_pickle(os.path.join(f'{base_directory}', jobs_info_filename))
        self.job_ids = list(self.jobs_info.keys())

        # Check running jobs status
        while len(self.job_ids) > 0:
            for job_id in self.job_ids:
                self.check_job_status(job_id, max_queuetime, max_runtime)
            
            # Save updated jobs info
            self.save_to_pickle(self.jobs_info, os.path.join(f'{base_directory}', jobs_info_filename))
            
            # Wait some time before checking again
            time.sleep(sleep_time)


    # Utility methods to save and load pickle files, may consider moving to class Utils

    @staticmethod
    def save_to_pickle(object, path_to_file):
        with open(path_to_file, 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(path_to_file):
        with open(path_to_file, 'rb') as handle:
            return pickle.load(handle)

