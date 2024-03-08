""" Definition of the Writers class

Writers of various input files (TODO: more documentation).

"""

import os

from ase import Atoms
from ase.io import write

from camus.structures import Structures

class Writers:

    def __init__(self, artn_parameters=None, lammps_parameters=None,
            sisyphus_parameters=None):
        """
        Initializes a new Writers object.

        Parameters:
            parameter: parameter description placeholder.
        """

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
            'verbose': '1',
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
        If `target_directory` is not given, `CAMUS_SISYPHUS_DATA_DIR` environment variable will be used.
        If self._artn_parameters is an empty dictionary, it will be automatically generated.

        Parameters:
            target_directory: directory in which to write the artn.in file

        """

        # Set the default target directory to CAMUS_SISYPHUS_DATA_DIR environment variable
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_SISYPHUS_DATA_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_SISYPHUS_DATA_DIR environment variable is not set.")

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


    def set_lammps_parameters(self, input_parameters=None, path_to_model=None, specorder=None, initial_sisyphus=False, minimization=False, **kwargs):
        """ Method that sets parameters to be written in lammps.in to self._lammps_parameters dictionary.

        Parameters:
            `input_parameters`: dictionary of LAMMPS commands and arguments in the format {'command1': 'args1' }.
            If `input_parameters` is not given, default parameters for a basic ARTn calculation will be used.
            `path to model`: path to the ML model to be used. If not specified, ML_MODEL environment variable will be used.
            `specorder`: order of atomic species. If not specified, a default hardcoded ordering will be used.
            `initial_sisyphus`: if True, standard parameters for a lammps.in file for a calculation of potential energy are set.
                                It is necessary to perform this calculation to perform the Sisyphus search.
            `minimization`: if True, standard parameters for a lammps.in file for minimization are set.

            WARNING: not specifying the above parameters may easily lead to wrong results.

            NOTE: this method is intended to be used mainly within the CAMUS algorithm for Sisyphus searches
            and therefore it currently doesn't have a lot of flexibility (dicts may not be the best choice).

        """

        if path_to_model is None:
            path_to_model = os.environ.get('ML_MODEL')
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
            'fix': '10 all artn alpha0 0.2 dmax 5.0',
            'timestep': '0.001',
            'reset_timestep': '0',
            'min_style': 'fire',
            'minimize': '1e-4 1e-5 5000 10000',
             }

        # Default parameters for the initial Sisyphus potential energy calculation
        default_initial_sisyphus = {
            'units': 'metal',
            'dimension': '3',
            'boundary': 'p p p',
            'atom_style': 'atomic',
            'atom_modify': 'map array',
            'read_data': 'lammps.data',
            'pair_style': 'allegro',
            'pair_coeff': f'* * {path_to_model} {specorder}',
            'thermo_style': 'custom step pe fnorm lx ly lz press pxx pyy pzz',
            'dump': 'initialDump all custom 1 initial_sisyphus_structure.xyz type x y z fx fy fz id',
            'dump_modify': 'initialDump sort id',
            'run': '0'
             }

        # Default parameters for a structure minimization
        default_minimization = {
            'units': 'metal',
            'dimension': '3',
            'boundary': 'p p p',
            'atom_style': 'atomic',
            'atom_modify': 'map array',
            'read_data': 'lammps.data',
            'pair_style': 'allegro',
            'pair_coeff': f'* * {path_to_model} {specorder}',
            'min_style': 'fire',
            'minimize': '1.0e-4 1.0e-6 100 1000',
            'dump': 'minimizeDump all custom 1 minimized.xyz type x y z fx fy fz id',
            'dump_modify': 'minimizeDump sort id',
            'run': 0}

        # If input_parameters aren't provided use default
        if input_parameters is not None:
            self._input_parameters = input_parameters
        else:
            input_parameters = default_parameters

        # If initial_sisyphus=True, use default_initial_sisyphus parameters
        if initial_sisyphus:
            input_parameters = default_initial_sisyphus

        # If minimization=True, use default_minimization parameters
        elif minimization:
            input_parameters = default_minimization

        # Set self._lammps_parameters
        self._lammps_parameters = input_parameters


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

    def set_sisyphus_parameters(self, **kwargs):
        """ Method that sets parameters to be written in sisyphus.sh to self._sisyphus_parameters dictionary.

        Parameters:
           dE_initial_threshold: Threshold for reaching the top of the hill [eV]
           dE_final_threshold: The search stops when E_current_minimum - E_initial <= dE_final_threshold (activated only after the top of the hill is passed) [eV]
           delr_threshold: At least one of the delr's has to be smaller than this, otherwise the search path is not connected and we restart ARTn [angstrom]
           maximum_steps: Maximum number of ARTn searches before we give up
           run_lammps: Command to run the LAMMPS executable (e.g. mpirun -np 1). If not given, defaults to the $RUN_LAMMPS environment variable. (Still requires testing)
           lammps_exe: Path to the LAMMPS executable. If not given, defaults to the $LAMMPS_EXE environment variable
           sisyphus_functions_path: Path to the sisyphus bash functions. If not given, defaults to $CAMUS_BASE/sisyphus_files/sisyphus_functions
           initial_lammps_in: Filename of the input file for initial LAMMPS minimization/PE calculation
           initial_lammps_out: Filename of the output file for initial LAMMPS minimization/PE calculation

        """
        default_parameters = {
            'dE_initial_threshold': '0.5',
            'dE_final_threshold': '0.1',
            'delr_threshold': '1.0',
            'maximum_steps': '100',
            'run_lammps': os.environ.get('RUN_LAMMPS'),
            'lammps_exe': os.environ.get('LAMMPS_EXE'),
            'lammps_flags': os.environ.get('LAMMPS_FLAGS'),
            'sisyphus_functions_path': os.path.join(os.environ.get('CAMUS_BASE'), 'sisyphus_files/sisyphus_functions'),
            'initial_lammps_in': 'initial_lammps.in',
            'initial_lammps_out': 'initial_lammps.out'
            }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self._sisyphus_parameters
        for key, value in default_parameters.items():
            self._sisyphus_parameters[key] = kwargs.pop(key, value)


    def write_sisyphus_script(self, target_directory=None, filename='sisyphus.sh'):
        """ Method that writes the main Sisyphus bash script to `target directory/filename` using self._sisyphus_parameters.
        If `target_directory` is not given, `CAMUS_SISYPHUS_DATA_DIR` environment variable will be used.
        If self._sisyphus_parameters is an empty dictionary, it will be automatically generated.

        Parameters:
            target_directory: directory in which to write the lammps.in file
            filename: name of the Sisyphus bash script

        """
        # Set the default target directory to CAMUS_SISYPHUS_DATA_DIR environment variable
        if target_directory is None:
            target_directory = os.environ.get('CAMUS_SISYPHUS_DATA_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_SISYPHUS_DATA_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.sisyphus_parameters:
            self.set_sisyphus_parameters()

        # Write the sisyphus.sh file to the target directory
        with open(os.path.join(target_directory, filename), 'w') as f:
            f.write(f"""#!/bin/bash

source {self.sisyphus_parameters['sisyphus_functions_path']}

{self.sisyphus_parameters['run_lammps']} {self.sisyphus_parameters['lammps_exe']} {self.sisyphus_parameters['lammps_flags']} -in {self.sisyphus_parameters['initial_lammps_in']} > {self.sisyphus_parameters['initial_lammps_out']} 

E_initial=$(grep "PotEng" initial_lammps.out -A1 | tail -1 | awk {{'print $2'}})

dE_initial=-1.0           # Difference between current minimum and initial energy [eV]
dE_initial_threshold={self.sisyphus_parameters['dE_initial_threshold']}  # Threshold for reaching the top of the hill [eV]
dE_final_threshold={self.sisyphus_parameters['dE_final_threshold']}    # The search stops when E_current minimum - E_initial <= dE_final_threshold (activated only after the top of the hill is passed) [eV]

delr_threshold={self.sisyphus_parameters['delr_threshold']}	  # At least one of the delr's has to be smaller than this, otherwise the search path is not connected and we restart ARTn [angstrom]

outer_counter=0           # Counting how many times an ARTn search is called (first call is labelled 0)
minima_counter=0          # Counting how many minimas (and saddle points) are accepted
basin_counter=0           # Counting how many times we started an ARTn search from the same configuration
maximum_steps={self.sisyphus_parameters['maximum_steps']}         # Maximum number of ARTn searches before we give up.

E_current=$E_initial      # Current configuration energy is initialized to the initial energy

define_file_names         # Define prefix "${{dE_initial_threshold}}_${{dE_final_threshold}}" for this run
cleanup_dat               # This will delete the *.dat files containing information of a previous run
cleanup_minima            # This will delete minimum_*.xyz, saddlepoint_*.xyz and LAMMPS data files from previous runs
initialize_files          # Create *.dat files for a fresh run



echo "$E_initial" >> $MINIMA_ENERGIES_FILE
echo "$E_initial" >> $ALL_ENERGIES_FILE


ACCEPTANCE_SIGN=">"       # First we want to climb up the hill
REJECTION_SIGN="<"        # First we want to climb up the hill


while (( $(echo "$dE_initial < $dE_initial_threshold" | bc -l) ))
do
    if (( $(echo "$outer_counter < $maximum_steps" | bc -l) ))
    then
        echo "MSG_$outer_counter: Running ARTn search #$outer_counter ..." >> $SISYPHUS_LOG_FILE
        {self.sisyphus_parameters['run_lammps']} {self.sisyphus_parameters['lammps_exe']} {self.sisyphus_parameters['lammps_flags']} -in lammps.in

        advance_search
    else
        echo "MSG_FAIL: Maximum number of searches ($maximum_steps) exceeded while climbing up the hill. Exiting." >> $SISYPHUS_LOG_FILE
        exit
    fi
done

echo "MSG_TOP: Top of the hill passed." >> $SISYPHUS_LOG_FILE
echo "MSG_TOP: Top of the hill minimum index: $minima_counter" >> $SISYPHUS_LOG_FILE

E_top=$E_current
Activation_E_forward=$(echo "$E_top - $E_initial" | bc -l)
echo "MSG_TOP: E_initial = $E_initial    E_top = $E_top    Activation_E_forward = $Activation_E_forward" >> $SISYPHUS_LOG_FILE


ACCEPTANCE_SIGN="<"       # Now we're going down the hill
REJECTION_SIGN=">"        # Now we're going down the hill


while (( $(echo "$dE_initial >= $dE_final_threshold" | bc -l) ))
do
    if (( $(echo "$outer_counter < $maximum_steps" | bc -l) ))
    then
        echo "MSG_$outer_counter: Running ARTn search #$outer_counter ..." >> $SISYPHUS_LOG_FILE
        {self.sisyphus_parameters['run_lammps']} {self.sisyphus_parameters['lammps_exe']} {self.sisyphus_parameters['lammps_flags']} -in lammps.in

        advance_search
    else
        echo "MSG_FAIL: Maximum number of searches ($maximum_steps) exceeded while going down the hill. Exiting." >> $SISYPHUS_LOG_FILE
        exit
    fi
done

echo "MSG_END: Threshold satisfied. Ending search." >> $SISYPHUS_LOG_FILE
echo "MSG_END: Final minimum index: $minima_counter" >> $SISYPHUS_LOG_FILE

E_final=$E_current
Delta_E_final_top=$(echo "$E_final - $E_top" | bc -l)
Delta_E_final_initial=$(echo "$E_final - $E_initial" | bc -l)

echo "MSG_END: E_initial = $E_initial    E_top = $E_top    E_final = $E_final" >> $SISYPHUS_LOG_FILE
echo "MSG_END: Activation_E_forward = $Activation_E_forward    Delta_E_final_top = $Delta_E_final_top    Delta_E_final_initial = $Delta_E_final_initial" >> $SISYPHUS_LOG_FILE""")


def write_lammps_data(input_structures, target_directory=None, prefixes='auto', specorder=None, write_masses=True,
        atom_style='atomic'):
    """ Creates LAMMPS data files from a list of ASE Atoms objects in a target directory.

    If `input_structures` is not given, self.structures are used.
    If `target_directory` is not given, CAMUS_LAMMPS_DATA_DIR environment variable is used.
    If `prefix='auto'`, a list of integers [0_, 1_, ...] will be used as prefixes of the file names.
    Otherwise, a list of prefixes should be given and the filenames will be `{prefix}lammps.data`.
    specorder: order of atom types in which to write the LAMMPS data file.
    If write_masses=True, Masses section will be added to the created file LAMMPS data file. In this case,
    """
    if input_structures is None:
        input_structures = self.structures
    if target_directory is None:
        target_directory = os.getenv('CAMUS_LAMMPS_DATA_DIR')

    # Create target directory if it does not exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Special case of single input structure:
    if isinstance(input_structures, Atoms): input_structures = [input_structures]

    # Write LAMMPS data files
    for i, structure in enumerate(input_structures):
        if prefixes == 'auto':
            prefix = str(i) + '_'
        else:
            if isinstance(prefixes, list):
                prefix = str(prefixes[i])
            elif isinstance(prefixes, str):
                prefix = prefixes

        file_name = os.path.join(target_directory, f'{prefix}lammps.data')
        write(file_name, structure, format='lammps-data', specorder=specorder, atom_style=atom_style)

        # Write masses if requested
        if write_masses:

            masses = []
            for spec in specorder:
                for atom in structure:
                    if atom.symbol == spec:
                        masses.append(atom.mass)
                        break
                    else: pass

            with open(file_name, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if line.strip() == 'Atoms':
                    lines.insert(i, 'Masses\n\n')
                    for j, spec in enumerate(specorder):
                        lines.insert(i+j+1, f'{j+1} {masses[j]} # {spec}\n')
                    lines.insert(i+j+2, '\n')
                    break

            with open(file_name, 'w') as f:
                f.writelines(lines)

    
def write_POSCAR(input_structure, specorder, target_directory=None):
    """ Writes a standard VASP input structure file (POSCAR), using an ASE Atoms object `input_structure`.
    The desired `specorder` (e.g. a list ['Br', 'I', 'Cs', 'Pb'] must be given to ensure 
    that the atomic species are reordered as in a POTCAR file.
    'target_directory' If not specified, `target_directory` defaults to the environment variable '$CAMUS_DFT_DIR'.
    
    """

    # Set default target_directory 
    if target_directory is None:
        target_directory = os.environ.get('CAMUS_DFT_DIR')
        if target_directory is None:
            raise ValueError("Target directory not specified and CAMUS_DFT_DIR environment variable is not set.")

    # Create target directory if it does not exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Reorder atoms
    sorted_atoms = Structures().sort_atoms(input_structure=input_structure, specorder=specorder) 

    # Write the POSCAR file to the target directory
    write(os.path.join(target_directory, 'POSCAR'), input_structure, format='vasp')
