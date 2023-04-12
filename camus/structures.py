""" Definition of the Structures class.

This module defines everything related to handling structures in the CAMUS algorithm. It is expected that the structures
are given as a list of ASE atoms objects.

"""

import random
import numpy as np
import os

from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ase.io import write
from ase.io.lammpsrun import read_lammps_dump

from dscribe.descriptors import ACSF
from dscribe.kernels import AverageKernel

class Structures:

    def __init__(self, structures=None, training_set=None, validation_set=None, test_set=None, 
sisyphus_set=None, minimized_set=None, descriptors=None, acsf_parameters=None):
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

        if structures is not None:
            self._structures = structures
        else:
            self._structures = []

        if training_set is not None:
            self._training_set = training_set
        else:
            self._training_set = []

        if validation_set is not None:
            self._validation_set = validation_set
        else:
            self._validation_set = []

        if test_set is not None:
            self._test_set = test_set
        else:
            self._test_set = []

        if sisyphus_set is not None:
            self._sisyphus_set = sisyphus_set
        else:
            self._sisyphus_set = []

        if minimized_set is not None:
            self._minimized_set = minimized_set
        else:
            self._minimized_set = []

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

    def write_lammps_data(self, target_directory=None, input_structures=None, prefixes='auto', specorder=None, write_masses=True,
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

    def set_charges(self, charges_input=None, input_structures=None):
        """ Method that takes a dictionary `charges_input = {atomic_type1: charge, atomic_type2: charge, ...}` and sets the charges to
        every structure in `input_structures` using ase.atoms.set_initial_charges() method. Can be useful for classical potential simulations.

        If `input_structures` is not given, self.structures are used.
        If `charges_input` are not given, defaults to charges given in https://doi.org/10.1039/D0TA03200J

        """
        #charges_input
        if charges_input is not None:
            self._charges_input = charges_input
        else:
            charges_input = {
                    'Pb': 0.9199,
                    'Cs': 1.0520,
                    'I': -0.6573,
                    'Br': -0.6573
                    }

        if input_structures is None:
            input_structures = self.structures

       # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]

        # Create the full charges array for all atoms and set_initial_charges(charges)
        for structure in input_structures:
            charges = np.empty(structure.get_global_number_of_atoms())

            for i, symbol in enumerate(structure.get_chemical_symbols()):
                charges[i] = charges_input[symbol]

            structure.set_initial_charges(charges)

    @staticmethod
    def parse_lammps_dump(specorder, log_lammps='log.lammps', dump_name='minimized.xyz'):
        """
        Reads a LAMMPS dump in `directory` and returns an ASE Atoms object with written forces and energies (which are read from `log_lammps`).
        """

        with open(dump_name) as f:
            structure = read_lammps_dump(f, specorder=specorder)

        # Get potential energy
        with open(log_lammps) as f:
            log_lines = f.readlines()

        for i, line in enumerate(log_lines):
            if 'Energy initial, next-to-last, final =' in line:
                energies_line = log_lines[i+1].strip()
                potential_energy = energies_line.split()[-1]
                structure.calc.results['energy'] = float(potential_energy)
                break

            elif 'Step PotEng' in line:
                energies_line = log_lines[i+1].strip()
                potential_energy = energies_line.split()[1]
                structure.calc.results['energy'] = float(potential_energy)
                break

        return structure

    @staticmethod
    def parse_sisyphus_xyz(filename, specorder):
        """
        Reads a '*initp.xyz' '*minimum*.xyz' or '*saddlepoint*xyz' file generated by Sisyphus and returns
        an ASE Atoms object with written forces and energy.
        """

        with open(filename) as f:
            lines = f.readlines()

        cell = np.array(lines[1].strip().split(' ')[1:10], dtype='float').reshape(3, 3, order='F')
        energy = float(lines[1].strip().split(':')[-1])
        positions = np.loadtxt(filename, skiprows=2, usecols=(1, 2, 3))
        forces = np.loadtxt(filename, skiprows=2, usecols=(4, 5, 6))
 
        atom_types = []
 
        for line in lines[2:]:
            atom_id = int(line.strip()[0])
            atom_types.insert(-1, specorder[atom_id - 1])
 
        atoms=Atoms(symbols=atom_types, positions=positions, pbc=True, cell=cell)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

        return atoms

    @staticmethod
    def sort_atoms(input_structure, specorder):
        """
        Sorts an ASE Atoms object by specorder. 
        """

        atomic_numbers = [Atom(sym).number for sym in specorder]
        atom_numbers = input_structure.get_atomic_numbers()
        order = np.argsort([atomic_numbers.index(n) for n in atom_numbers])
        sorted_atoms = input_structure[order]

        return sorted_atoms

    """

    Model accuracy:
    
    """
    @staticmethod
    def model_accuracy(dft_structures, lammps_structures, energy_limit=0.1, force_limit=0.2):
    """
    [Notes on model_accuracy]
    """
        # check the datasets are the same lenghts
        if len(dft_structures) != len(lammps_structures):
            raise ValueError("The datasets cannot be compared if they don't contain the same number of structures.")

        # initiate self.lists
        dft_energies = []
        dft_forces = []
        lammps_energies = []
        lammps_forces = []

        # Calculate potential energies and forces:
        for dft_structure in dft_structures:
            dft_energies.append(dft_structure.get_potential_energy())
            dft_forces.append(dft_structure.get_forces())

        for lammps_structure in lammps_structures:
            lammps_energies.append(lammps_structure.get_potential_energy())
            lammps_forces.append(lammps_structure.get_forces())

        for i, lammps_structure in enumerate(lammps_structures):
            if abs((dft_energies[i] - lammps_energies[i])/dft_energies[i]) > energy_limit:
                energies_over_limit_indices.append(i)
                energies_over_limit.append(lammps_structure[i])

        for i, lammps_structure in enumerate(lammps_structures):
            dft_force = np.array(dft_forces[i])
            lammps_force = np.array(lammps_forces[i])
            diffrence = np.where(np.any(abs((dft_force-lammps_force)/dft_force)>force_limit))[0]
            if len(difference) > 0:
                forces_over_limit_indices.append(i)
                forces_over_limit.append(lammps_structure[i])
        
        # create union lists
        structures_over_limit_indices = list(set().union(energies_over_limit_indices, forces_over_limit_indices))
        structures_over_limit = list(set().union(energies_over_limit, forces_over_limit))

        # return list of indices and list of ASE Atoms
        return structures_over_limit_indices, structures_over_limit
