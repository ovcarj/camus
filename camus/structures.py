""" Definition of the Structures class.

This module defines everything related to handling structures in the CAMUS algorithm. It is expected that the structures
are given as a list of ASE atoms objects.

"""

import random
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

import camus.tools.utils

from functools import cached_property

from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump

from dscribe.descriptors import ACSF

from collections import Counter


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
            self.structures = structures
        else:
            self.structures = []

        if training_set is not None:
            self.training_set = training_set
        else:
            self.training_set = []

        if validation_set is not None:
            self.validation_set = validation_set
        else:
            self.validation_set = []

        if test_set is not None:
            self.test_set = test_set
        else:
            self.test_set = []

        if sisyphus_set is not None:
            self.sisyphus_set = sisyphus_set
        else:
            self.sisyphus_set = []

        if minimized_set is not None:
            self.minimized_set = minimized_set
        else:
            self.minimized_set = []

        if descriptors is not None:
            self.descriptors = descriptors
        else:
            self.descriptors = np.empty(0)

        if acsf_parameters is not None:
            self.acsf_parameters = acsf_parameters
        else:
            self.acsf_parameters = {}
        
        ###
        self.self_energies = {}
        
    @cached_property
    def chemical_symbols(self):
        """ Creates a list of chemical symbols for each structure in the
        Structures instance. Note: group_by_composition() should be updated accordingly.
        """

        symbols = []

        for structure in self.structures:
            symbols.append(structure.get_chemical_symbols())

        return symbols


    def set_acsf_parameters(self, **kwargs):
        """ Method that sets parameters to be used for creating the ACSF descriptors in self.acsf_parameters dictionary.

        Parameters:
            parameter: parameter description placeholder

        """
        default_parameters= {
            'rcut': 6.0,
            'g2_params': [(1, 2), (1, 4), (1, 8),(1,16)], #[(1, 2), (1, 4)],
            'g3_params': [1,2],
            'g4_params': [(1, 4, 4), (1, 4, -1), (1, 8, 4), (1, 8, -1)],#[(1, 2, 1), (1, 4, 1)], 
            'species': ['Cs', 'Pb', 'Br', 'I'],
            'sparse': False,
            'periodic': True
            }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self.acsf_parameters
        for key, value in default_parameters.items():
            self.acsf_parameters[key] = kwargs.pop(key, value)


    def calculate_descriptors(self, input_structures=None, positions=None, n_jobs=1):

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
            sparse=self.acsf_parameters['sparse'],
            periodic=self.acsf_parameters['periodic']
            )

        # Special case of single input structure:
        #if isinstance(input_structures, Atoms): input_structures = [input_structures]

        #for atoms in input_structures:
        #    descriptor = acsf_descriptor.create(atoms)
        #    self.descriptors.append(descriptor)
        self.descriptors = acsf_descriptor.create(input_structures, positions=positions, n_jobs=n_jobs)


    def group_by_composition(self, input_structures=None, specorder=None):
        """
        Groups structures into self.structures_grouped_by_composition dictionary of form
        {(composition): {Structures(structure_group), indices: indices in original Structures.structures object}}
        """

        if input_structures is None:
            input_structures = self.structures

        # Get the unique chemical symbols present in the structures
        if specorder is None:
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
        self.structures_grouped_by_composition = {count: {'structures': camus.tools.utils.new_list(), 'indices': camus.tools.utils.new_list()} 
                for count in unique_counts}

        for i, structure in enumerate(input_structures):
            counts = tuple(structure_counts[symbol][i] for symbol in chemical_symbols)
            self.structures_grouped_by_composition[counts]['structures'].append(structure)
            self.structures_grouped_by_composition[counts]['indices'].append(i)

        # Change the groups structures to Structures instances
        for count, group in self.structures_grouped_by_composition.items():
            self.structures_grouped_by_composition[count]['structures'] = Structures(group['structures'])


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


    def set_charges(self, charges_input=None, input_structures=None):
        """ Method that takes a dictionary `charges_input = {atomic_type1: charge, atomic_type2: charge, ...}` and sets the charges to
        every structure in `input_structures` using ase.atoms.set_initial_charges() method. Can be useful for classical potential simulations.

        If `input_structures` is not given, self.structures are used.
        If `charges_input` are not given, defaults to charges given in https://doi.org/10.1039/D0TA03200J

        """
        #charges_input
        if charges_input is not None:
            self.charges_input = charges_input
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
        Get displacements between reference structure (given by `reference_index`) and all other structures.
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


    def get_displacements_per_type(self, reference_index=0, use_IRA=False):
        """
        Get displacements between reference structure (given by `reference_index`) and all other structures
        separated by chemical species into a dictionary.
        """

        chemical_symbols = self.chemical_symbols
        displacements = self.get_displacements(reference_index=reference_index, use_IRA=use_IRA)
        type_indices_dict = camus.tools.utils.create_index_dict(chemical_symbols[reference_index])
        
        displacements_per_type = {key: {'indices': indices} for key, indices in type_indices_dict.items()}
        for key in displacements_per_type.keys():
            displacements_per_type[key]['displacements'] = np.array([displacement[displacements_per_type[key]['indices']] for displacement in displacements])

        return displacements_per_type


    def get_average_displacement_per_type(self, reference_index=0, use_IRA=False):
        """
        Calculates the average displacement for each chemical species between `reference` structure (given by index) and all other structures.
        No prealignment is performed.
        """

        displacements_per_type = self.get_displacements_per_type(reference_index=reference_index, use_IRA=use_IRA)
        average_displacement_per_type = {chemical_type: np.average(value['displacements'], axis=1) for chemical_type, value in displacements_per_type.items()}

        return average_displacement_per_type


    def get_average_displacement_all_structures(self, reference_index=0, use_IRA=False):
        """
        Calculates the average displacement from a `reference` structure for each chemical species 
        across all structures.
        """

        average_displacement_per_type = self.get_average_displacement_per_type(reference_index=reference_index, use_IRA=use_IRA)
        average_displacement_all_structures = {chemical_type: np.average(value) for chemical_type, value in average_displacement_per_type.items()}

        return average_displacement_all_structures


    def get_maximum_displacements_per_type(self, reference_index=0, use_IRA=False):
        """
        Calculates the maximum displacements for each chemical species between `reference` structure (given by index) and all other structures.
        No prealignment is performed.
        """

        displacements_per_type = self.get_displacements_per_type(reference_index=reference_index, use_IRA=use_IRA)
        maximum_displacements_per_type = {chemical_type: {'maximum_displacements': np.max(value['displacements'], axis=1), 
            'atomic_index': np.array([displacements_per_type[chemical_type]['indices'][m] for m in np.argmax(value['displacements'], axis=1)])} 
            for chemical_type, value in displacements_per_type.items()}

        return maximum_displacements_per_type


    def get_maximum_displacement_all_structures(self, reference_index=0, use_IRA=False):
        """
        Calculates the maximum of all displacements for each chemical species between `reference` structure (given by index) and all other structures.
        No prealignment is performed.
        """

        maximum_displacements_per_type = self.get_maximum_displacements_per_type(reference_index=reference_index, use_IRA=use_IRA)
        maximum_displacement_all_structures = {chemical_type: {'maximum_displacement': np.max(value['maximum_displacements']), 
            'atomic_index': value['atomic_index'][np.argmax(value['maximum_displacements'])],
            'structure_index': np.argmax(value['maximum_displacements'])} 
            for chemical_type, value in maximum_displacements_per_type.items()}

        return maximum_displacement_all_structures


    def plot_displacements(self, reference_index=0, savefig=False, save_format='pdf', dpi=400, fname='displacements', return_fig_ax=False):
        """
        Plots average and maximum displacements from a reference structure for a Structures object.
        For now, testing if returning fig, ax makes sense (for other plots).
        """

        # Get data

        average_displacement_per_type = self.average_displacement_per_type
        average_displacement_all_structures = self.average_displacement_all_structures
        maximum_displacements_per_type = self.maximum_displacements_per_type
        maximum_displacement_all_structures = self.maximum_displacement_all_structures

        chemical_species = average_displacement_per_type.keys()
        no_of_structures = len(self.structures) - 1  #Won't plot reference structure

        all_indices = np.arange(0, no_of_structures + 1, 1)
        relevant_indices = np.delete(all_indices, np.where(all_indices == reference_index))

        # Plot definition, colors, bar widths

        fig, ax = camus.tools.utils.create_plot(xlabel='', ylabel=r'Displacement $(\AA)$', fontsize=18)

        total_width = 0.75
        width = total_width / no_of_structures

        # Iterate over chemical species and number of structures

        for i, species in enumerate(chemical_species):

            for j, index in enumerate(relevant_indices):

                x = i + 1 + total_width * (j / no_of_structures - 0.5)

                color1 = 'royalblue'
                color2 = 'salmon'

                bar_maximum = ax.bar(x, maximum_displacements_per_type[species]['maximum_displacements'][index], 
                        width=width, color=color2, align='edge', edgecolor='black', linewidth=0.7, label='Maximum displacement')
                bar_average = ax.bar(x, average_displacement_per_type[species][index], 
                        width=width, color=color1, align='edge', edgecolor='black', linewidth=0.7, label='Average displacement')

        # Define xticks, legend, etc...

        xticks = list(chemical_species)
        plt.xticks(np.arange(1, len(xticks) + 1), xticks, fontsize=18)
        ax.minorticks_on()
        ax.tick_params(axis='y', direction='in', which='both', labelsize=15, length=8)
        ax.tick_params(axis='y', which='minor', length=4)
        ax.tick_params(axis='x', which='both', bottom=False)

        ax.legend(handles=[bar_average, bar_maximum], loc='best', fontsize=15)
        
        # Save plot

        if savefig == False:
            plt.show()
        elif savefig == True:
            plt.savefig(fname=fname + '.' + save_format, format=save_format, bbox_inches='tight', dpi=dpi)

        if return_fig_ax:
            return fig, ax


    # Testing if it makes sense to put these methods as cached properties, maybe it will be convenient
    displacements = cached_property(get_displacements)
    displacements_per_type = cached_property(get_displacements_per_type)
    average_displacement_per_type = cached_property(get_average_displacement_per_type)
    average_displacement_all_structures = cached_property(get_average_displacement_all_structures)
    maximum_displacements_per_type = cached_property(get_maximum_displacements_per_type)
    maximum_displacement_all_structures = cached_property(get_maximum_displacement_all_structures)

    ###
    """
    def energy_correction(self, input_structures=None, specorder=None):

        if input_structures is None:
            input_structures = self.structures

        # If `specorder` not given assume the specorder from the first structure in `input_structures`
        if specorder is not None:
            specorder = specorder
        else:
            specorder = []
            for species in input_structures[0].get_chemical_symbols():
                if species not in specorder:
                    specorder.append(species)

        # Calculate self energies if you haven't before but in this case `input_structures` can't be  only one structure
        if not self.self_energies:
            self.calculate_self_energies()
        
        # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]
        
        corrected_energies = []
        for structure in input_structures:
           
            chemical_species = structure.get_chemical_symbols()
            count = Counter()

            for species in chemical_species:
                count[species] += 1

            corrected_energy = structure.get_potential_energy()
            for species in specorder:
                corrected_energy -= count[species] * self.self_energies[species]
            corrected_energies.append(corrected_energy)

        return np.array(corrected_energies)
    """


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


def calculate_self_energies(input_structures, specorder=None):

    # The fit needs enough structures to be `good`
    if isinstance(input_structures, Atoms):
        raise RuntimeError('You cannot fit on just one structure')
    elif len(input_structures) < 100:
        warnings.warn(f'{len(input_structures)} might not be enough structures to produce good fit.\n Condsider increasing the size of your dataset.', FutureWarning)
    else:
        pass

    # If `specorder` not given assume the specorder from the first structure in `input_structures`
    if specorder is not None:
        specorder = specorder
    else:
        specorder = []
        for species in input_structures[0].get_chemical_symbols(): 
            if species not in specorder:
                specorder.append(species)

    chemical_species_counter = {species: [] for species in specorder}

    # Count the number of atoms of each species in the `input_structures`
    for i, structure in enumerate(input_structures):

        chemical_species = structure.get_chemical_symbols()
        count = Counter()

        for species in chemical_species:
            count[species] += 1

        for species, count in count.items():
            chemical_species_counter[species].append(count)

        # If given species is not present in the specific structure add count of zero
        for species in chemical_species_counter:
            if len(chemical_species_counter[species]) != i + 1:
                chemical_species_counter[species].append(0)

    # The total number of atoms of each species in the `input_structures`
    no_per_species = [chemical_species_counter[species] for species in specorder]
    no_per_species.append([1] * (len(input_structures)))  # Corrected line 
    no_per_species = np.array(no_per_species).transpose()

    # Get energy for each structure 
    energy = [structure.get_potential_energy() for structure in input_structures]
    energy = np.array(energy)

    # Least square fit
    lstsq_solution, _, _, _ = np.linalg.lstsq(no_per_species, energy, rcond=None)
    interaction_energy = lstsq_solution[-1] # if it's even useful to us

    # Create `self_energies` dictionary for comprehensive output
    self_energies = {}
    for species, energy in zip(specorder, lstsq_solution[:-1]):
        self_energies[species] = energy

    return self_energies


def self_energy_correction(input_structures, self_energies, specorder=None):
    """
    - This method takes a list of ASE Atoms objects, corrects their energies using the previously calcualted `self_energies` and returns another list of    ASE Atoms objects which now have the `corrected_energy`
    """
    # Special case of single input structure:
    if isinstance(input_structures, Atoms): input_structures = [input_structures]

    corrected_structures = []
    for input_structure in input_structures:
        composition = camus.tools.utils.create_index_dict(input_structure.get_chemical_symbols())
        corrected_energy = input_structure.get_potential_energy()

        for spec in specorder:
            corrected_energy -= len(composition[spec])*self_energies[spec]

        forces = input_structure.get_forces()
        input_structure.calc = SinglePointCalculator(input_structure, energy=corrected_energy, forces=forces)
        corrected_structures.append(input_structure)

    return corrected_structures


def evaluate_sisyphus_run(base_directory=None, eval_dictionary_path=None, specorder=None, dataset=None, write_extxyz=False, concatenate=False, energy_min=1.0, energy_max=2.0, traj_filename='retraining_set.extxyz'):
    """
    - `base_directory` the same as for DFT
    - `dataset` for fitting of self energies, ideally the one used for training 
    - `write=False` to initially calculate and save the corrected energies into the `eval_dictionary` to be used to determine the correct parameters for retraining set choice
    """

    if base_directory is not None:
        base_directory = base_directory
    else:
        base_directory = os.path.join(os.getcwd(), 'base_dft')
        
    if eval_dictionary_path is None:
        eval_dictionary_path = os.path.join(base_directory, 'eval_dictionary.pkl')

    eval_dictionary = camus.tools.utils.load_pickle(eval_dictionary_path)

    #
    self_energies = calculate_self_energies(input_structures=dataset, specorder=specorder)

    structures_for_retraining = []

    for key in eval_dictionary.keys():

        # self energy correction
        self_energy_correct = 0
        composition = eval_dictionary[key]['composition']
        for spec in self_energies.keys():
            no_of_spec = len(composition[spec])
            self_energy_correct += no_of_spec*self_energies[spec]

        corrected_dft_energy = eval_dictionary[key]['dft_energy'] - self_energy_correct
        corrected_ml_energy = eval_dictionary[key]['ml_energy'] - self_energy_correct

        eval_dictionary[key]['corrected_dft_energy'] = abs(corrected_dft_energy)
        eval_dictionary[key]['corrected_ml_energy'] = abs(corrected_ml_energy)

        # Energy evaluation
        energy_difference = abs(corrected_dft_energy - corrected_ml_energy)
        if energy_difference <= energy_min:
            eval_dictionary[key]['retraining_flag'] = 'G'
        elif energy_difference <= energy_max:
            eval_dictionary[key]['retraining_flag'] = 'R'
        else:
            eval_dictionary[key]['retraining_flag'] = 'D'

        # Collect all structures marked 'R'
        if eval_dictionary[key]['retraining_flag'] == 'R':
            outcar_file = os.path.join(eval_dictionary[key]['dft_directory'], 'OUTCAR')
            structure = read(outcar_file)
            structures_for_retraining.append(structure)

    # Check if we found any structures for retraining
    if write_extxyz and len(structures_for_retraining) != 0:
        # Write out the dataset
        if concatenate:
            original = read(dataset)
            original.append(structures_for_retraining)

            #self energy correction
            corrected_original = self_energy_correction(input_structures=original, self_energies=self_energies, specorder=specorder)    
    
            original_filename = os.path.basename(dataset)
            updated_filename = os.path.splitext(original_filename)[0] + '_update' + os.path.splitext(original_filename)[1]
            write(os.path.join(base_directory, updated_filename), corrected_original, format='extxyz')
        else:
            #self energy correction
            corrected_structures_for_retraining = self_energy_correction(input_structures=structures_for_retraining, self_energies=self_energies, specorder=specorder)
            write(os.path.join(base_directory, traj_filename), corrected_structures_for_retraining, format='extxyz')

    # Save the dictionary
    camus.tools.utils.save_to_pickle(eval_dictionary, os.path.join(base_directory, 'eval_dictionary.pkl'))


def plot_evaluation(path_to_eval_dictionary, xlabel=r'$E_{ML} - E_{self}$', ylabel=r'$E_{DFT} - E_{self}$', sizex=15.0, sizey=15.0, fontsize=15, fit=False, histogram=False):
    """
    `fit` might make sense to show some sort of progress over several runs
    `histogram` to make the assesment of the correct energy interval for retraing easier
    """

    eval_dictionary = camus.tools.utils.load_pickle(path_to_eval_dictionary)
    ml_energies = []
    dft_energies = []
    for key in eval_dictionary.keys():
        ml_energies.append(eval_dictionary[key]['corrected_ml_energy'])
        dft_energies.append(eval_dictionary[key]['corrected_dft_energy'])

    if not histogram:
        fig, ax = camus.tools.utils.create_plot(xlabel=xlabel, ylabel=ylabel, sizex=sizex, sizey=sizey, fontsize=fontsize)
        ax.scatter(ml_energies, dft_energies, s=50, zorder=0)
        # Plot a E_ML = E_DFT line
        e_max = max(ml_energies)
        x = [0,e_max]
        y = [0,e_max]
        ax.plot(x,y, color='red', label="E_ML=E_DFT", zorder=2)

        if fit:
            fit_coefficients = np.polyfit(ml_energies, dft_energies, 1)
            fit_line = np.poly1d(fit_coefficients)
            ax.plot(ml_energies, fit_line(ml_energies), color='red', label="Linear fit", zorder=4)
            
        ax.legend(loc='best', fontsize=15)
        plt.show()

    else:
        energy_diffs = []
        x = []
        for i, ml_energy in enumerate(ml_energies):
            energy_diff = abs(ml_energy - dft_energies[i])
            energy_diffs.append(energy_diff)
            x.append(i)
        
        total_width = 0.75
        width = total_width / len(x)

        fig, ax = camus.tools.utils.create_plot(xlabel='', ylabel=r'Energy difference [eV]', fontsize=18)
        bar_diff = ax.bar(x, energy_diffs, width=width, color='royalblue', align='edge', edgecolor='black', linewidth=0.7, label="|E_DFT-E_ML|")

        ax.tick_params(axis='y', direction='in', which='both', labelsize=15, length=8)
        ax.tick_params(axis='y', which='minor', length=4)

        ax.legend(handles=[bar_diff], loc='best', fontsize=15)
        plt.show()

