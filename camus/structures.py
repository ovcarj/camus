""" Definition of the Structures class.

This module defines everything related to handling structures in the CAMUS algorithm. It is expected that the structures
are given as a list of ASE atoms objects.

"""

import random
import numpy as np
import os
import glob

from functools import cached_property

from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ase.io import write
from ase.io.lammpsrun import read_lammps_dump

from dscribe.descriptors import ACSF
from dscribe.kernels import AverageKernel

from collections import Counter

from camus.utils import save_to_pickle, load_pickle

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
            'g2_params': [(1, 2), (1, 4)], #[(1, 2), (1, 4), (1, 8),(1,16)],
            'g3_params': [1,2],
            'g4_params': [(1, 2, 1), (1, 4, 1)], #[(1, 4, 4), (1, 4, -1), (1, 8, 4), (1, 8, -1)],
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

    @staticmethod
    def group_by_composition(input_structures, specorder):

        if specorder is None:
            # Get the unique chemical symbols present in the structures
            chemical_symbols = set()
            for structure in input_structures:
                chemical_symbols.update(structure.get_chemical_symbols())
        else:
            chemical_symbols = specorder

        # Create a dictionary to store the counts of each chemical symbol in each structure
        structure_counts = {symbol: [] for symbol in chemical_symbols}

        # Populate the structure_counts dictionary
        for structure in input_structures:
            counts = Counter(structure.get_chemical_symbols())
            for symbol in chemical_symbols:
                structure_counts[symbol].append(counts.get(symbol, 0))

        # Identify the unique combinations of counts
        unique_counts = set(map(tuple, zip(*structure_counts.values())))

        # Filter the structures based on the unique count combinations
        structures_grouped_by_composition = {count: [] for count in unique_counts}
        for i, structure in enumerate(input_structures):
            counts = tuple(structure_counts[symbol][i] for symbol in chemical_symbols)
            structures_grouped_by_composition[counts].append(structure)

        return structures_grouped_by_composition

    @staticmethod
    def find_unique_structures_by_composition(reference_set_structures, candidate_set_structures, threshold=0.90, metric='laplacian', gamma=1, specorder=None):
        '''
        find groups for reference set
        find groups for candidate set
        create a `common_compositions` dictionary {composition: reference_structures[], candidate[structures]}
        if candidate set group not in reference_groups keys -> candidate `automatically unique`
        find_unique_structures within the `common_compositions` dictionary
        append all the `unique` structures into a single set of `unique_structures`
        '''

        # Create grouped datasets
        reference_groups = Structures.group_by_composition(reference_set_structures, specorder=specorder)
        candidate_groups = Structures.group_by_composition(candidate_set_structures, specorder=specorder)
        
        # Create sets of element compositions 
        reference_compositions = set(reference_groups.keys())
        candidate_compositions = set(candidate_groups.keys())
        common_compositions = reference_compositions & candidate_compositions

        # Initiate `common_compostion_groups` dictionary
        common_composition_groups = {}
        # Cycle through the compositions the two have in common and assign a reference and candidate set to each composition
        for key in common_compositions:
            common_composition_groups[key] = {
                'reference_structures': reference_groups[key],
                'candidate_structures': candidate_groups[key],
            }
       
        unique_structures = []
        
        # If candidate not in common_groups then automatically considered `unique`
        for key in candidate_compositions:
            if key not in common_compositions:
                unique_structures.extend(candidate_groups[key])

        # Run `find_unique_structures` function within each composition group 
        for key in common_compositions:
            unique = Structures.find_unique_structures(reference_set_structures = Structures(common_composition_groups[key]['reference_structures']), candidate_set_structures = Structures(common_composition_groups[key]['candidate_structures']), threshold=threshold, metric=metric, gamma=gamma)
            if unique != []:
                unique_structures.extend(unique)
        
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
        Reads an '*initp.xyz', '*minimum*.xyz' or '*saddlepoint*xyz' file generated by Sisyphus and returns
        an ASE Atoms object with written forces and energy.
        """

        with open(filename) as f:
            lines = f.readlines()

        sorted_lines = sorted(lines[2:], key=lambda line: int(line.split()[-1]))

        cell = np.array(lines[1].strip().split(' ')[1:10], dtype='float').reshape(3, 3, order='F')
        energy = float(lines[1].strip().split(':')[-1])

        positions_list = []
        forces_list = []
        for sorted_line in sorted_lines:
            positions_list.append(np.array([float(i) for i in sorted_line.strip().split()[1:4]]))
            forces_list.append(np.array([float(i) for i in sorted_line.strip().split()[4:7]]))

        positions = np.array(positions_list)
        forces = np.array(forces_list)

        atom_types = []

        for line in sorted_lines:
            atom_id = int(line.strip()[0])
            atom_types.append(specorder[atom_id - 1])

        atoms = Atoms(symbols=atom_types, positions=positions, pbc=True, cell=cell)
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

    def get_displacements(self, reference_index=0, use_IRA=False):
        """
        Get displacements between `reference` structure (given by index) and all other structures.
        """

        displacements_list = []
        reference_structure = self.structures[reference_index] 
        
        for i, structure in enumerate(self.structures):

            if use_IRA:
                permutations, distances = calculate_displacement_IRA(reference_structure, structure)
                displacements_list.append(distances)

            else:
                distances = calculate_displacement(reference_structure, structure)
                displacements_list.append(distances)

        return np.array(displacements_list)

    displacements = cached_property(get_displacements)



class STransition():

    def __init__(self, stransition=None, base_directory=None, sisyphus_dictionary_path=None, calculation_label=None, **kwargs):
        """
        Initializes a new STransition object which stores and analyzes all information from a single transition obtained from a Sisyphus calculation.
        `stransition` is assumed to be a (key, value) pair from `sisyphus_dictionary.pkl`.

        If `stransition` is not given, an entry from `sisyphus_dictionary.pkl` is read from `sisyphus_dictionary_path` (this is mainly for testing purposes). 
        f `sisyphus_dictionary_path` is not given, it's searched for in `base_directory`.
        If `base_directory` is not given, CWD is assumed.

        Parameters:
            to_be_written (TODO): TODO.
        """

        if stransition is not None:
            self.stransition_label, stransition_info = stransition

        else:
            if base_directory is not None:
                self._base_directory = base_directory
            else:
                self._base_directory = os.getcwd()

            if sisyphus_dictionary_path is not None:
                self._sisyphus_dictionary_path = sisyphus_dictionary_path
            else:
                self._sisyphus_dictionary_path = os.path.join(f'{self._base_directory}', 'sisyphus_dictionary.pkl')

            self._sisyphus_dictionary = load_pickle(self._sisyphus_dictionary_path)
            
            if calculation_label is not None:
                self._stransition_label = calculation_label
            else:
                self._stransition_label = list(self._sisyphus_dictionary.keys())[0]

            stransition_info = self._sisyphus_dictionary[self._stransition_label]

        for key in stransition_info.keys():

            if ('structures' not in key):
                setattr(self, key, stransition_info[key])

            else:
                setattr(self, key, Structures(stransition_info[key]))

        self.activation_e_forward = np.max(self.all_energies) - self.all_energies[0]  #added this in case maximum saddlepoint_e < maximum minimum_e

    # cached_property used intentionally as an example
    # calculates energies for each transition (saddlepoint_n - minimum_n)
    @cached_property
    def small_transition_energies(self):
        energies = self.saddlepoints_energies - self.minima_energies[:len(self.saddlepoints_energies)]
        return energies




"""
Various helper functions start here
"""


def calculate_displacement_IRA(atoms1, atoms2):
    """
    Calculates per-atom displacement between two ASE atoms objects using the IRA method.
    See https://github.com/mammasmias/IterativeRotationsAssignments
    """

    try:
        import ira_mod

        ira = ira_mod.IRA()
    
        nat1, nat2 = len(atoms1), len(atoms2)
        symbols1, symbols2 = atoms1.get_chemical_symbols(), atoms2.get_chemical_symbols()
        positions1, positions2 = atoms1.get_positions(), atoms2.get_positions()
        cell = np.array(atoms1.get_cell())
    
        permutations, distances = ira.cshda(nat1=nat1, nat2=nat2, coords1=positions1, coords2=positions2, lat=cell)
    
        return permutations, distances

    except ImportError:
        print('IRA not found (install from https://github.com/mammasmias/IterativeRotationsAssignments)')



def calculate_displacement(atoms1, atoms2):
    """
    Calculates per-atom displacement between two ASE atoms.
    """

    # Get the cell parameters (assumes same cell between structures)
    cell = atoms1.get_cell()

    # Apply PBC to the coordinates of atoms2 relative to atoms1
    displacement_vectors = atoms2.get_positions(wrap=True) - atoms1.get_positions(wrap=True)

    # Apply minimum image convention to handle periodicity
    displacement_vectors -= np.round(displacement_vectors.dot(np.linalg.inv(cell))) @ cell

    displacement_magnitude = np.linalg.norm(displacement_vectors, axis=1)

    return displacement_magnitude
