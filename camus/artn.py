""" Definition of the ARTn class.

This module defines everything related to handling ARTn inputs and outputs.

"""

import os

class ARTn:

    def __init__(self, artn_outputs=[], artn_parameters={}):
        """
        Initializes a new ARTn object.

        Parameters:
            artn_outputs (list of ARTn output filenames): TODO: user should be able to parse these outputs.
                Defaults to an empty list.
        """

        self._artn_outputs = artn_outputs
        self._artn_parameters = artn_parameters

    @property
    def artn_outputs(self):
        return self._artn_outputs

    @artn_outputs.setter
    def artn_outputs(self, new_artn_outputs):
        self._artn_outputs = new_artn_outputs

    @artn_outputs.deleter
    def artn_outputs(self):
        del self._artn_outputs

    @property
    def artn_parameters(self):
        return self._artn_parameters

    @artn_parameters.setter
    def artn_parameters(self, new_artn_parameters):
        self._artn_parameters = new_artn_parameters

    @artn_parameters.deleter
    def artn_parameters(self):
        del self._artn_parameters


    def set_artn_parameters(self, **kwargs):
        """ Method that sets parameters to be written in artn.in to self._artn_parameters dictionary.

        Parameters:
            parameter: parameter description placeholder

        """
        default_parameters= {
            'engine_units': '\'lammps/metal\'',
            'verbose': '0',
            'zseed': None,
            'lrestart': '.false.',
            'ninit': '2',
            'lpush_final': '.true.',
            'nsmooth': '2',
            'forc_thr': '0.05',
            'push_step_size': '0.1',
            'push_mode': '\'all\'',
            'lanczos_disp': '1.0D-3',
            'lanczos_max_size': '16',
            'eigen_step_size': '0.15',
            'frelax_ene_thr': None }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self._artn_parameters
        for key, value in default_parameters.items():
            self._artn_parameters[key] = kwargs.pop(key, value)


    def write_artn_in(self, target_directory=None):
        """ Method that writes a standard artn.in file to a `target directory` using self._artn_parameters.
        If `target_directory` is not given, `CAMUS_ARTN_DATA_DIR` environment variable will be used.
        If self._artn_parameters is an empty dictionary, it will be automatically generated.

        Parameters:
            target_directory: directory in which to write the artn.in file

        """

        # Set the default target directory to CAMUS_ARTN_DATA_DIR environment variable
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_ARTN_DATA_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_ARTN_DATA_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.artn_parameters:
            self.set_artn_parameters()

        # Define the artn.in file content
        artn_in_content = "&ARTN_PARAMETERS\n"
        for key, value in self.artn_parameters.items():
            if value is not None:
                artn_in_content += f"  {key} = {self._artn_parameters[key]}\n"
        artn_in_content += "/\n"

        # Write the artn.in file to the target directory
        with open(os.path.join(target_directory, 'artn.in'), 'w') as f:
            f.write(artn_in_content)

