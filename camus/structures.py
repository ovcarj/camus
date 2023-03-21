""" Definition of the Structures class.

This module defines everything related to handling structures in the CAMUS algorithm. It is expected that the structures
are given as a list of ASE atoms objects.

"""

import random
import numpy as np
import os
from ase import Atoms
from ase.io import write
from dscribe.descriptors import ACSF
from dscribe.kernels import AverageKernel

class Structures:

    def __init__(self, structures=[], training_set=[], validation_set=[], test_set=[], 
sisyphus_set=[], minimized_set=[], descriptors=None, acsf_parameters=None):
        """
        Initializes a new Structures object with a list of structures and optional training, validation and test sets.

        Parameters:
            structures (list of ASE Atoms objects): A general list of structures to be stored in the Structures object.
            training_set (list of ASE Atoms objects): List of structures to be used as the training set.
                Defaults to an empty list.
            validation_set (list of ASE Atoms objects): List of structures to be used as the validation set.
                Defaults to an empty list.
            test_set (list of ASE Atoms objects): List of structures to be used as the test set.
                Defaults to an empty list.
            sisyphus_set (list of ASE Atoms objects): List of structures to be used as input structures for Sisyphus searches.
            minimized_set (list of ASE Atoms objects): Container for minimized structures.
                Defaults to an empty list.
        """

        self._structures = structures
        self._training_set = training_set
        self._validation_set = validation_set
        self._test_set = test_set
        self._sisyphus_set = sisyphus_set
        self._minimized_set = minimized_set 

        # Check https://stackoverflow.com/questions/73887738/dict-instance-variable-being-changed-for-all-instances
        # https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil/
        # Should refactor the code to take care of all instances of this not-a-bug-but-a-feature brilliancy...

        if descriptors is not None:
            self._descriptors = descriptors
        else:
            self._descriptors = []

        if acsf_parameters is not None:
            self._acsf_parameters = acsf_parameters
        else:
            self._acsf_parameters = {}

    @property
    def structures(self):
        return self._structures

    @structures.setter
    def structures(self, new_structures):
        self._structures = new_structures

    @structures.deleter
    def structures(self):
        del self._structures

    @property
    def training_set(self):
        return self._training_set

    @training_set.setter
    def training_set(self, new_structures):
        self._training_set = new_structures

    @training_set.deleter
    def training_set(self):
        del self._training_set

    @property
    def validation_set(self):
        return self._validation_set

    @validation_set.setter
    def validation_set(self, new_structures):
        self._validation_set = new_structures

    @validation_set.deleter
    def validation_set(self):
        del self._validation_set

    @property
    def test_set(self):
        return self._test_set

    @test_set.setter
    def test_set(self, new_structures):
        self._test_set = new_structures

    @test_set.deleter
    def test_set(self):
        del self._test_set

    @property
    def sisyphus_set(self):
        return self._sisyphus_set

    @sisyphus_set.setter
    def sisyphus_set(self, new_structures):
        self._sisyphus_set = new_structures

    @sisyphus_set.deleter
    def sisyphus_set(self):
        del self._sisyphus_set

    @property
    def minimized_set(self):
        return self._minimized_set

    @minimized_set.setter
    def minimized_set(self, new_structures):
        self._minimized_set = new_structures

    @minimized_set.deleter
    def minimized_set(self):
        del self._minimized_set

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, new_descriptors):
        self._descriptors = new_descriptors

    @descriptors.deleter
    def descriptors(self):
        del self._descriptors

    @property
    def acsf_parameters(self):
        return self._acsf_parameters

    @acsf_parameters.setter
    def acsf_parameters(self, new_acsf_parameters):
        self._acsf_parameters = new_acsf_parameters

    @acsf_parameters.deleter
    def acsf_parameters(self):
        del self._acsf_parameters

    def set_acsf_parameters(self, **kwargs):
        """ Method that sets parameters to be used for creating the ACSF descriptors in self._acsf_parameters dictionary.

        Parameters:
            parameter: parameter description placeholder

        """
        default_parameters= {
            'rcut': 6.0,
            'g2_params': [(1, 2), (1, 4), (1, 8),(1,16)],
            'g3_params': [1,2],
            'g4_params': [(1, 4, 4), (1, 4, -1), (1, 8, 4), (1, 8, -1)],
            'species': ['Cs', 'Pb', 'Br', 'I'],
            'periodic': True
            }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self._acsf_parameters
        for key, value in default_parameters.items():
            self._acsf_parameters[key] = kwargs.pop(key, value)

    def calculate_descriptors(self, input_structures=None):

        if input_structures is None:
            input_structures = self.structures

        # Set parameters if the user didn't set them explicitly beforehand
        if not self.acsf_parameters:
            self.set_acsf_parameters()

        acsf_descriptor = ACSF(
            rcut=self.acsf_parameters['rcut'],
            g2_params=self.acsf_parameters['g2_params'],
            g3_params=self.acsf_parameters['g3_params'],
            g4_params=self.acsf_parameters['g4_params'],
            species=self.acsf_parameters['species'],
            periodic=self.acsf_parameters['periodic']
            )

        for atoms in input_structures:
            descriptor = acsf_descriptor.create(atoms)
            self.descriptors.append(descriptor)

    @staticmethod
    def find_unique_structures(reference_set_structures, candidate_set_structures, threshold=0.90, metric='laplacian', gamma=1):
        # reference_set_structures and candidate_set_structures must be instances of Structures class

        if not candidate_set_structures.descriptors:
            candidate_set_structures.calculate_descriptors()

        if not reference_set_structures.descriptors:
            reference_set_structures.calculate_descriptors()

        unique_structures = []
        similarity_of_structures = []

        for i, candidate_descriptor in enumerate(candidate_set_structures.descriptors):
            is_unique = True

            for reference_descriptor in reference_set_structures.descriptors:
                ak = AverageKernel(metric=metric, gamma=gamma) 
                ak_kernel = ak.create([reference_descriptor, candidate_descriptor])
                similarity = ak_kernel[0, 1]
                similarity = round(similarity, 5) # arbitrary 5
                if similarity >= threshold:
                    is_unique = False
                    break

            if is_unique:
                unique_structures.append(candidate_set_structures.structures[i])

        return unique_structures

    def create_datasets(self, input_structures=None, training_percent=0.8, validation_percent=0.1, test_percent=0.1, randomize=True):
        """ Separates the structures from `input_structures` into training, validation and test sets.
        If `input_structures` is not given, self.structures will be used.

        If randomize=True, randomizes the ordering of structures.
        """
        if input_structures is None:
            input_structures = self.structures

        if training_percent + validation_percent + test_percent != 1.0:
            raise ValueError("Percentages do not add up to 1.0")

        if len(input_structures) == 0:
            raise ValueError("No structures to create datasets from.")

        structures = input_structures.copy()

        if randomize:
            random.shuffle(structures)

        num_structures = len(structures)
        num_train = int(num_structures * training_percent)
        num_val = int(num_structures * validation_percent)
        num_test = num_structures - num_train - num_val

        self.training_set = structures[:num_train]
        self.validation_set = structures[num_train:num_train+num_val]
        self.test_set = structures[num_train+num_val:]

    def create_sisyphus_set(self, input_structures=None, mode='random', indices=None, number_of_structures=5):
        """ Creates a `number_of_structures` of structures to be used as input structures for Sisyphus searches.

        If the optional list of `input_structures` is not provided, use the Structures object's own `structures`
        attribute as the base set from which to create the structures.

        If `mode='random'`, randomly select `number_of_structures` from the base set.
        If `mode='indices'`, select structures indexed by `indices` from the base set.
        """

        if input_structures is None:
            base_set = self.structures
        else:
            base_set = input_structures

        if mode == 'random':
            self.sisyphus_set = random.sample(base_set, number_of_structures)

        elif mode == 'indices':
            self.sisyphus_set = [base_set[i] for i in indices]

        else:
            raise ValueError("Unsupported mode. Choose 'random' or 'indices'.")

    def get_energies_and_forces(self, input_structures=None):
        """ Read the energies and forces for a set of structures.

        If the optional argument `structures` is not provided,
        use the Structures object's own `structures` attribute.
        """

        if not input_structures:
            structures = self.structures
        else:
            structures = input_structures

        energies = []
        forces = []

        for structure in structures:
            energy = structure.get_potential_energy()
            force = structure.get_forces()

            energies.append(energy)
            forces.append(force)

        return np.array(energies), np.array(forces)

    def write_lammps_data(self, target_directory=None, input_structures=None, prefixes='auto', specorder=None, write_masses=False,
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

    def set_charges(self, charges_input={}, input_structures=None):
        """ Method that takes a dictionary `charges_input = {atomic_type1: charge, atomic_type2: charge, ...}` and sets the charges to
        every structure in `input_structures` using ase.atoms.set_initial_charges() method. Can be useful for classical potential simulations.

        If `input_structures` is not given, self.structures are used.
        If `charges_input` are not given, defaults to charges given in https://doi.org/10.1039/D0TA03200J

        """
        if input_structures is None:
            input_structures = self.structures

        if not charges_input:
            charges_input = {
                    'Pb': 0.9199,
                    'Cs': 1.0520,
                    'I': -0.6573,
                    'Br': -0.6573
                    }

        # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]

        # Create the full charges array for all atoms and set_initial_charges(charges)
        for structure in input_structures:
            charges = np.empty(structure.get_global_number_of_atoms())

            for i, symbol in enumerate(structure.get_chemical_symbols()):
                charges[i] = charges_input[symbol]

            structure.set_initial_charges(charges)
