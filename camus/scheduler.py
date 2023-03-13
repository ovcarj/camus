""" Definition of the Scheduler class.

This module defines everything related to handling cluster scheduling.

"""

import os
from abc import ABC, abstractmethod

class Scheduler(ABC):

    def __init__(self, scheduler_parameters={}):
        """
        Initializes a new Scheduler object.

        Parameters:
            scheduler_parameters: parameters to be written in the sub.sh file.

        """

        self._scheduler_parameters = scheduler_parameters

    @abstractmethod
    def set_scheduler_parameters(self, **kwargs):
        ...

    @abstractmethod
    def write_submission_script(self, target_directory, filename):
        ...

    @abstractmethod
    def run_submission_script(self, filename):
        ...

class Slurm(Scheduler):


    def __init__(self, scheduler_parameters={}):

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

        # Write the sisyphus.sh file to the target directory
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

    def run_submission_script(self, filename):
        ...


