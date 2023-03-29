""" Definition of the Scheduler class.

This module defines everything related to handling cluster scheduling.

"""

import os
import subprocess
import time

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

        default_parameters= {
            'partition': 'normal',
            'job-name': 'sisyphus',
            'output': '%x-%j.out',
            'error': '%x-%j.err',
            'mem': '7gb', 
            'nodes': '1',
            'ntasks': '1',
            'run_command': f'{run_lammps} {lammps_exe} -in {input_file} > {output_file}' 
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
#SBATCH --job-name={self.scheduler_parameters['job-name']}
#SBATCH --output={self.scheduler_parameters['output']}
#SBATCH --error={self.scheduler_parameters['error']}
#SBATCH --mem={self.scheduler_parameters['mem']}
#SBATCH --nodes={self.scheduler_parameters['nodes']}
#SBATCH --ntasks={self.scheduler_parameters['ntasks']}

module purge
module load gnu9
module load openmpi4/4.1.1

export MKL_CBWR="AVX2"
export I_MPI_FABRICS=shm:ofi
ulimit -s unlimited

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64

{self.scheduler_parameters['run_command']}
""")

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
                'submission_time': submission_time, 'start_time': 0, 'queue_time': 0, 'run_time': 0}


    def check_job_status(self, job_id, max_queuetime, max_runtime):
        """ Checks whether a job is queueing, running or failed.
            If max_queuetime or max_runtime (in seconds) has ellapsed, cancel the job.
        """

        current_time = time.time()

        squeue_result = subprocess.check_output(['squeue', '-h', '-j', str(job_id)], stderr=subprocess.DEVNULL)

        # Job completed (not running anymore)
        if len(squeue_result.strip()) == 0:
            self.job_ids.remove(job_id)
            self.jobs_info[f'{job_id}']['job_status'] = 'FINISHED'

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
            elif self.jobs_info[f'{job_id}']['job_status'] in ['BF', 'CA', 'DL', 'F', 'NF', 'PR', 'ST', 'TO']:
                self.job_ids.remove(job_id)
                subprocess.run(["scancel", job_id])

            # Job running
            elif self.jobs_info[f'{job_id}']['job_status'] == 'R':

                # Check if this is the first instance of seeing the job running
                if self.jobs_info[f'{job_id}']['run_time'] == 0:
                    self.jobs_info[f'{job_id}']['start_time'] = current_time

                self.jobs_info[f'{job_id}']['run_time'] = current_time - self.jobs_info[f'{job_id}']['start_time']

                # Job running too long
                if self.jobs_info[f'{job_id}']['run_time'] > max_runtime:
                    self.jobs_info[f'{job_id}']['job_status'] = 'MAX_RUN_TIME_ELLAPSED'
                    self.job_ids.remove(job_id)
                    subprocess.run(["scancel", job_id])

            # Transient job status - hopefully nothing special is happening
            else: pass

