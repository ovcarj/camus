""" Definition of the ARTn class.

This module defines everything related to handling ARTn inputs and outputs.

To be discussed - should Sisyphus related methods be moved to the
Camus class, or should this class simply be renamed Sisyphus, as it deals
with more general things than just ARTn? The second option seems more logical...
"""

import os

class ARTn:

    def __init__(self, artn_outputs=[], artn_parameters={}, lammps_parameters={},
            sisyphus_parameters={}):
        """
        Initializes a new ARTn object.

        Parameters:
            artn_outputs (list of ARTn output filenames): TODO: user should be able to parse these outputs.
                Defaults to an empty list.
        """

        self._artn_outputs = artn_outputs
        self._artn_parameters = artn_parameters
        self._lammps_parameters = lammps_parameters
        self._sisyphus_parameters = sisyphus_parameters

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
        self._artn_parameters = {} # Probably makes more sense this way...
        self._artn_parameters = new_artn_parameters

    @artn_parameters.deleter
    def artn_parameters(self):
        del self._artn_parameters

    @property
    def lammps_parameters(self):
        return self._lammps_parameters

    @lammps_parameters.setter
    def lammps_parameters(self, new_lammps_parameters):
        self._lammps_parameters = {} # Probably makes more sense this way...
        self._lammps_parameters = new_lammps_parameters

    @lammps_parameters.deleter
    def lammps_parameters(self):
        del self._lammps_parameters
        
    @property
    def sisyphus_parameters(self):
        return self._sisyphus_parameters

    @sisyphus_parameters.setter
    def sisyphus_parameters(self, new_sisyphus_parameters):
        self._sisyphus_parameters = {} # Probably makes more sense this way...
        self._sisyphus_parameters = new_sisyphus_parameters

    @sisyphus_parameters.deleter
    def sisyphus_parameters(self):
        del self._sisyphus_parameters


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

    def set_lammps_parameters(self, input_parameters={}, path_to_model=None, specorder=None, initial_sisyphus=False, **kwargs):
        """ Method that sets parameters to be written in lammps.in to self._lammps_parameters dictionary.

        Parameters:
            `input_parameters`: dictionary of LAMMPS commands and arguments in the format {'command1': 'args1' }.
            If `input_parameters` is not given, default parameters for a basic ARTn calculation will be used.
            `path to model`: path to the ML model to be used. If not specified, a default hardcoded directory will be used.
            `specorder`: order of atomic species. If not specified, a default hardcoded ordering will be used.
            `initial_sisyphus`: if True, a standard lammps.in file for a calculation of potential energy is created.
                                It is necessary to perform this calculation to perform the Sisyphus search.

            WARNING: not specifying the above parameters may easily lead to wrong results.

            NOTE: this method is intended to be used mainly within the CAMUS algorithm for Sisyphus searches 
            and therefore it currently doesn't have a lot of flexibility (dicts may not be the best choice).

        """

        if path_to_model is None:
            path_to_model = '/storage/perovskites/allegro/deploys/model0'
        if specorder is None:
            specorder = 'Br I Cs Pb'

        # Default parameters for ARTn search
        default_parameters= {
            'units': 'metal',
            'dimension': '3',
            'boundary': 'p p p',
            'atom_style': 'atomic',
            'atom_modify': 'map array',
            'read_data': 'lammps.data',
            'pair_style': 'allegro',
            'pair_coeff': f'* * {path_to_model} {specorder}',
            'fix': '10 all artn alpha0 0.2 dmax 5.0 0.1',
            'timestep': '0.001',
            'reset_timestep': '0',
            'min_style': 'fire',
            'minimize': '1e-4 1e-5 5000 10000',
             }

        # Default parameters for the initial Sisyphus calculation
        default_initial_sisyphus = {
            'units': 'metal',
            'dimension': '3',
            'boundary': 'p p p',
            'atom_style': 'atomic',
            'atom_modify': 'map array',
            'read_data': 'lammps.data',
            'pair_style': 'allegro',
            'pair_coeff': f'* * {path_to_model} {specorder}',
            'compute': 'eng all pe/atom',
            'compute': 'eatoms all reduce sum c_eng',
            'thermo': '100',
            'thermo_style': 'custom step pe fnorm lx ly lz press pxx pyy pzz c_eatoms',
            'run': '0'
             }


        # If input parameters are not given, use default_parameters
        if not input_parameters:
            input_parameters = default_parameters

        # If initial_sisyphus=True, use default_initial_sisyphus parameters
        if initial_sisyphus:
            input_parameters = default_initial_sisyphus

        # Set self._lammps_parameters
        for key, value in input_parameters.items():
            self._lammps_parameters[key] = value


    def write_lammps_in(self, target_directory=None, filename='lammps.in'):
        """ Method that writes a lammps.in file to `target directory/filename` using self._lammps_parameters.
        If `target_directory` is not given, `CAMUS_LAMMPS_DATA_DIR` environment variable will be used.
        If self._lammps_parameters is an empty dictionary, it will be automatically generated.

        Parameters:
            target_directory: directory in which to write the lammps.in file
            filename: name of the lammps.in file

        """

        # Set the default target directory to CAMUS_LAMMPS_DATA_DIR environment variable
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_LAMMPS_DATA_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_LAMMPS_DATA_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.lammps_parameters:
            self.set_lammps_parameters()

        # Define the lammps.in file content
        lammps_in_content = "clear\n"
        for key, value in self.lammps_parameters.items():
            if value is not None:
                if isinstance(value, list):
                    for v in value:
                        lammps_in_content += f"{key}       {v}\n"
                else:
                    lammps_in_content += f"{key}       {value}\n"

        # Write the lammps.in file to the target directory
        with open(os.path.join(target_directory, filename), 'w') as f:
            f.write(lammps_in_content)

    def write_sisyphus_script(self, target_directory=None, filename='sisyphus.sh'):
        """ Method that writes the main Sisyphus bash script to `target directory/filename` using self._sisyphus_parameters.
        If `target_directory` is not given, `CAMUS_ARTN_DATA_DIR` environment variable will be used.
        If self._sisyphus_parameters is an empty dictionary, it will be automatically generated.

        Parameters:
            target_directory: directory in which to write the lammps.in file
            filename: name of the Sisyphus bash script


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
        if not self.sisyphus_parameters:
            self.set_sisyphus_parameters()


