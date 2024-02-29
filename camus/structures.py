""" Definition of the Structures class.

This module defines everything related to handling structures in the CAMUS algorithm. It is expected that the structures
are given as a list of ASE atoms objects.

"""

import random
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import camus.utils

from functools import cached_property

from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump

from dscribe.descriptors import ACSF
from dscribe.kernels import AverageKernel

from collections import Counter

import warnings

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
            self.descriptors = []

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
        self.structures_grouped_by_composition = {count: {'structures': camus.utils.new_list(), 'indices': camus.utils.new_list()} 
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
        type_indices_dict = camus.utils.create_index_dict(chemical_symbols[reference_index])
        
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

        fig, ax = camus.utils.create_plot(xlabel='', ylabel=r'Displacement $(\AA)$', fontsize=18)

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

class STransition():

    def __init__(self, stransition=None, base_directory=None, sisyphus_dictionary_path=None, calculation_label=None, **kwargs):
        """
        Initializes a new STransition object which stores and analyzes all information from a single transition obtained from a Sisyphus calculation.
        `stransition` is assumed to be a (key, value) pair from `sisyphus_dictionary.pkl`.

        If `stransition` is not given, an entry from `sisyphus_dictionary.pkl` is read from `sisyphus_dictionary_path` (this is mainly for testing purposes). 
        f `sisyphus_dictionary_path` is not given, it's searched for in `base_directory`.
        If `base_directory` is not given, CWD is assumed.

        List of available attributes:
            activation_e_forward, directory, small_transition_energies, all_energies, minima_energies, 
            status, basin_counters, minima_structures, transition_structures, 
            delta_e_final_initial, saddlepoints_energies, delta_e_final_top, saddlepoints_structures

        Parameters:
            to_be_written (TODO): TODO.
        """

        if stransition is not None:
            self._stransition_label, stransition_info = stransition

        else:
            if base_directory is not None:
                self._base_directory = base_directory
            else:
                self._base_directory = os.getcwd()

            if sisyphus_dictionary_path is not None:
                self._sisyphus_dictionary_path = sisyphus_dictionary_path
            else:
                self._sisyphus_dictionary_path = os.path.join(f'{self._base_directory}', 'sisyphus_dictionary.pkl')

            self._sisyphus_dictionary = camus.utils.load_pickle(self._sisyphus_dictionary_path)
            
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
    """
    Calculates the energy for each transition (saddlepoint_n - minimum_n)
    """
    @cached_property
    def small_transition_energies(self):
        energies = self.saddlepoints_energies - self.minima_energies[:len(self.saddlepoints_energies)]
        return energies

    def plot_stransition(self, function=None, **kwargs):
        '''
        customizable parameters:
            -
        functions:
            - `basin_counters` ... will additionally plot the number of times ARTn had to be restarted from each minimum
            - `small_transition_energies` ... will additionally plot the height of each `small transition` along the STransition path 
            - `displacement`
        '''
    
        initial_structure = self.all_energies[0]
        initial_structure -= initial_structure
    
        transition_structure = np.max(self.all_energies)
        transition_structure -= self.all_energies[0] 
    
        minima_energies = self.minima_energies
        minima_energies -= self.all_energies[0] 
    
        saddlepoints_energies = self.saddlepoints_energies
        saddlepoints_energies -= self.all_energies[0] 
    
        all_energies = self.all_energies 
        all_energies -= self.all_energies[0] 
        all_indices = range(len(all_energies))
    
        transition_index, = np.where(all_energies == transition_structure)
        transition_index = int(transition_index)
    
        minima_indices, saddlepoints_indices = [], []
        for a in all_indices: minima_indices.append(a) if a%2 == 0 else saddlepoints_indices.append(a)
        
        # Parameter_dict
        default_parameters = {
            'xlabel': None,
            'ylabel': None,
            'plot_title': None,
            'size_xy': (float(10), float(7.5)),
            'annotate_x': float(),
            'annotate_y': float(),
            'size_of_marker': float(65),
            'fontsize': 15,
            'color': [
                'gray', #throughline
                'cyan', #minima
                'black',  #saddles
                'limegreen', #initial
                'blue',  #transition
                'red' #thresholds
                ],
            'xticks': all_indices,
            'legend': False,
            'save_as': None
            }
    
        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)
    
        plot_parameters = {}
        for key, value in default_parameters.items():
            plot_parameters[key] = kwargs.pop(key, value)
    
        # Set up the plot
        fig, ax = plt.subplots(figsize=(plot_parameters['size_xy']))
        # Throughline
        ax.plot(all_indices, all_energies, color=plot_parameters['color'][0], ls='--', zorder=1)
        # Points along the `STransition` path
        ax.scatter(0, initial_structure, color=plot_parameters['color'][3], s=plot_parameters['size_of_marker'], marker='s',zorder=2, label='Initial structure')
        ax.scatter(minima_indices[1:], minima_energies[1:], color=plot_parameters['color'][1], s=plot_parameters['size_of_marker'], zorder=2)
        ax.scatter(saddlepoints_indices[:int(transition_index/2)], saddlepoints_energies[:int(transition_index/2)], color=plot_parameters['color'][2], s=plot_parameters['size_of_marker'], zorder=2)
        ax.scatter(transition_index, transition_structure, color=plot_parameters['color'][4], s=plot_parameters['size_of_marker']+50, marker='p', zorder=2, label='Transition state')
        ax.scatter(saddlepoints_indices[int(transition_index/2+1):], saddlepoints_energies[int(transition_index/2+1):], color=plot_parameters['color'][2], s=plot_parameters['size_of_marker'], zorder=2)
    

        # Threshold lines
        ax.axhline(y=abs(self.delta_e_final_top), ls='dotted', color=plot_parameters['color'][5], zorder=0, label='Top/Bottom threshold')
        ax.axhline(y=self.delta_e_final_initial, ls='dotted', color=plot_parameters['color'][5], zorder=0)
        # Axes
        ax.set_xlabel(plot_parameters['xlabel'], fontsize=plot_parameters['fontsize'])
        ax.set_ylabel(plot_parameters['ylabel'], fontsize=plot_parameters['fontsize'])
        ax.tick_params(direction='in', which='both', labelsize=plot_parameters['fontsize'])
        ax.set_xticks(ticks=plot_parameters['xticks']) #which indices to plot along the axis
    
        if plot_parameters['legend']:
            plt.legend(fontsize=plot_parameters['fontsize']-2)
    
        if plot_parameters['plot_title'] is not None:
            plt.title(plot_parameters['plot_title'], fontsize=plot_parameters['fontsize'])

        if plot_parameters['plot_title'] is not None:
            plt.title(plot_parameters['plot_title'], fontsize=plot_parameters['fontsize'])

        # Plot additional info
        if function is not None:
            if function == 'basin_counters':
                transition_property = self.basin_counters 
                property_indices = minima_indices[:-1]
    
            if function == 'small_transition_energies':
                transition_property = self.small_transition_energies 
                property_indices = saddlepoints_indices
    
            for i, prop in enumerate(transition_property):
                ax.annotate(prop, (property_indices[i], all_energies[property_indices[i]]), xytext=(property_indices[i] + plot_parameters['annotate_x'], all_energies[property_indices[i]] + plot_parameters['annotate_y']), size = plot_parameters['fontsize']-3)
    
        # Save if you wish to
        if plot_parameters['save_as'] is not None:
            fig.savefig(fname=f"sisplot.{plot_parameters['save_as']}", bbox_inches='tight', format=plot_parameters['save_as'])
    
        plt.show()


    def write_report(self, report_name=None, target_directory=None):

        if report_name is not None:
            report_name = report_name
        else:
            report_name = 'sisyphus.report'

        if target_directory is None:
            target_directory = os.environ.get('CAMUS_SISYPHUS_ANALYSIS_DIR')
            if target_directory is None:
                raise ValueError("Target directory not specified and CAMUS_SISYPHUS_ANALYSIS_DIR environment variable is not set.")

        # Create target directory if it does not exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Intro & shameless plug
        report_content = "#Sisyphus report file generated by...\n\n"
        report_content += "\t CCCCC   AAAAA  MM   MM UU   UU  SSSSS \n"
        report_content += "\tCCCCCCC AAAAAAA MMM MMM UU   UU SSSSSSS\n"
        report_content += "\tCC      AA   AA MMMMMMM UU   UU SSS    \n"
        report_content += "\tCC      AA   AA MMMMMMM UU   UU  SSSSS \n"
        report_content += "\tCC      AAAAAAA MM M MM UU   UU     SSS\n"
        report_content += "\tCCCCCCC AAAAAAA MM   MM UUUUUUU SSSSSSS\n"
        report_content += "\t CCCCC  AA   AA MM   MM  UUUUU   SSSSS \n\n"

        # Header
        label = self.directory.strip().split('sis_')[-1]
        no_of_structures = len(self.transition_structures.structures)
        activation_e_forward = self.activation_e_forward
        status = self.status
        directory = self.directory
        divider_line = '_' * 120

        report_content += f"stransition_label: {label}\tno_of_structures: {no_of_structures}\tactivation_e_forward: {activation_e_forward}\tstatus: {status}\t\n\ndirectory: {directory}\n\n"
        report_content += f"{divider_line}\n\n"

        #Initial calculation parameters
        report_content += "#Initial calculation parameters\n\n"

        for key, value in self.initial_parameters.items():
            report_content += f"{key}: {value}\n"
        
        report_content += "\n"
        
        # Specorder & composition
        count = Counter(self.transition_structures.chemical_symbols[0])
        specorder = ''
        composition = ''

        for key, value in count.items():
            specorder += f"{key.strip()} "
            composition += f"{key.strip()}{value}"

        report_content += f"specorder: {specorder}\ncomposition: {composition}\n\n"

        # Potential used
        report_content += f"potential: {self.potential}\n\n"
        
        report_content += f"{divider_line}\n\n"

        # Results
        report_content += "#Resutls of the run\n\n"

        delta_e_final_initial = self.delta_e_final_initial
        delta_e_final_top = self.delta_e_final_top

        report_content += f"delta_e_final_initial/top: {delta_e_final_initial}\t{delta_e_final_top}\n\n"
 
        # Displacements
        report_content += "average/maximum displacement:\n"
        report_content += f"{'species' : <20}{'disp_avg' : <8}{'disp_max' : ^35}{'str_idx' : ^5}\n"
        for species in count.keys():
            disp_max = self.transition_structures.maximum_displacement_all_structures[species]['maximum_displacement']
            str_idx = self.transition_structures.maximum_displacement_all_structures[species]['structure_index']
            disp_avg = self.transition_structures.average_displacement_per_type[species][str_idx]

            report_content += f"{species : ^10}{disp_avg : ^25}{disp_max : ^22}{str_idx : >10}\n"

        report_content += "\n-------STRUCTURES-------\n\n"

        # Structures
        report_content += f"{'structure_no': <20}{'type' : <8}{'energy' : ^35}{'STE' : ^5}{'basin_count' : >20}\n"

        str_type = ['I']
        while len(str_type) < no_of_structures:
            str_type.append('S')
            str_type.append('M')

        top_energy = np.max(self.all_energies)
        top_index = int(np.where(self.all_energies == top_energy)[0])
        str_type[top_index] = 'T'

        small_trans_energies = []
        for energy in [round(e,5) for e in self.small_transition_energies]:
            small_trans_energies.append('-')
            small_trans_energies.append(energy)
        while len(small_trans_energies) < no_of_structures:
            small_trans_energies.append('-')

        basins = list(self.basin_counters)
        basin_count = []
        for basin in basins:
            basin_count.append(int(basin))
            basin_count.append('-')
        while len(basin_count) < no_of_structures:
            basin_count.append('-')

        for i in range(no_of_structures):
            report_content += f"{i : ^10} {str_type[i] : ^22} {round(self.all_energies[i],5) : ^20} {small_trans_energies[i] : ^20} {basin_count[i] : >5}\n"

        # Write it down
        with open(os.path.join(target_directory, report_name), 'w') as r:
            r.write(report_content)

class STransitions():

    def __init__(self, sisyphus_dictionary=None, base_directory=None, sisyphus_dictionary_path=None, 
            sisyphus_analysis_directory=None, save_analysis=False, **kwargs):
        """
        Initializes a new STransitions object which stores and analyzes all information from all Sisyphus calculations in a `sisyphus_dictionary.pkl`.
        Analysis can be saved to `sisyphus_analysis_directory` if `save_analysis` == True.

        `sisyphus_dictionary` should be the loaded `sisyphus_dictionary.pkl`.

        If `stransitions_pickle` is not given, `sisyphus_dictionary.pkl` is tried to be read from `sisyphus_dictionary_path`. 
        f `sisyphus_dictionary_path` is not given, `sisyphus_dictionary.pkl` is searched for in `base_directory`.
        If `base_directory` is not given, CWD is assumed.

        Parameters:
            to_be_written (TODO): TODO.
        """

        # Read Sisyphus dictionary
        if sisyphus_dictionary is not None:
            self._sisyphus_dictionary = sisyphus_dictionary

        else:
            if base_directory is not None:
                self._base_directory = base_directory
            else:
                self._base_directory = os.getcwd()

            if sisyphus_dictionary_path is not None:
                self._sisyphus_dictionary_path = sisyphus_dictionary_path
                self._sisyphus_dictionary = camus.utils.load_pickle(sisyphus_dictionary_path)

            else:
                self._sisyphus_dictionary_path = os.path.join(f'{self._base_directory}', 'sisyphus_dictionary.pkl')

            self._sisyphus_dictionary = camus.utils.load_pickle(self._sisyphus_dictionary_path)
            
        # Get Sisyphus analysis directory path
        if sisyphus_analysis_directory is not None:
            self._sisyphus_analysis_directory = sisyphus_analysis_directory

        else:
            self._sisyphus_analysis_directory = os.environ.get('CAMUS_SISYPHUS_ANALYSIS_DIR')


        # Initialize self.stransitions attribute as a dictionary of STransitions objects

        self.stransitions = {key: STransition(stransition=(key, value)) for key, value in self._sisyphus_dictionary.items() if value['status']=='PASSED' or value['status'].startswith('FAILED_')}

        # Initialize self.stransitions_properties dictionary

        self.stransitions_properties = {key: {} for key in self.stransitions.keys()} 


    def get_stransition_property(self, calculation_label, stransition_property):
        """
        Convenience method that copies STransition object property to a dictionary of form {calculation_label: {property: value}}.
        """

        # Special case for displacements
        displacement_types = ['average_displacement_all_structures', 'average_displacement_per_type', 'displacements', 'displacements_per_type', 
                'maximum_displacement_all_structures', 'maximum_displacements_per_type']

        structure_types = ['minima_structures', 'saddlepoints_structures', 'transition_structures']

        if (stransition_property == 'displacements'):

            for structure_type in structure_types:

                for displacement_type in displacement_types:
                
                    property_name = f'{structure_type.split("_")[0]}_{displacement_type}'

                    self.stransitions_properties[calculation_label].update({property_name: 
                            self.stransitions[calculation_label].__getattribute__(structure_type).__getattribute__(displacement_type)})

        # Standard case
        else:
            
            self.stransitions_properties[calculation_label].update({stransition_property: 
                self.stransitions[calculation_label].__getattribute__(stransition_property)})


    def get_all_properties(self):
        """
        Stores all properties of STransition to self.stransitions_properties 
        (except for structures; but stores displacements).
        """

        calculation_labels = [key for key in self.stransitions.keys()]

        attributes = [attribute for attribute in dir(self.stransitions[calculation_labels[0]])
                if (not attribute.startswith('_') and not 'structures' in attribute)]

        for calculation_label in calculation_labels:

            for attribute in attributes:

                self.get_stransition_property(calculation_label, attribute)

            self.get_stransition_property(calculation_label, 'displacements')


    def map_stransitions(self, structure_type='transition'):
        """
        Concatenates all self.stransitions[*calc_label*].`structure_type`_structures into
        a single self.concatenated_structures object.
        Creates a dictionary {calc_label: [indices in self.concatenated_stransitions]} to self.concatenated_map.
        Creates the inverse dictionary {index in self.concatenated_stransition: calc_label} for convenience to self.inverse_map.
        
        `structure_type` can be ['transition', 'minima', 'saddlepoints'].
        """

        structures_type = f'{structure_type}_structures'
        concatenated_structures = []
        calculation_labels = self.stransitions.keys()

        self.concatenated_map = {calculation_label: camus.utils.new_list for calculation_label in calculation_labels}
        self.inverse_map = camus.utils.new_dict()

        structure_counter = 0
        
        for calculation_label in calculation_labels:

            structures = self.stransitions[calculation_label].__getattribute__(structures_type).structures
            no_of_structures = len(structures)

            indices = np.arange(structure_counter, structure_counter + no_of_structures)
            self.concatenated_map[calculation_label] = indices
            structure_counter += no_of_structures

            for original_index, new_index in enumerate(indices):

                self.inverse_map[new_index]= {'calculation_label': calculation_label, 'original_index': original_index}

            concatenated_structures.append(structures)

        concatenated_structures_flat = [item for sublist in concatenated_structures for item in sublist]

        self.concatenated_structures = Structures(concatenated_structures_flat)


    def prefilter_stransitions(self, specorder, mode=None, structure_type='transition', displaced_elements=None, activation_energy_threshold=None):
        """
        Placeholder for a method which could possibly preprocess STransitions before it goes to 
        structure comparisons.
        Returns a dictionary {calc_label: flags, ...} which should indicate if some structures
        should be skipped.

        Maybe we could put prefiltered structures into self.prefiltered_stransitions 
        and then use structure_type arguments in map_stransitions and comparisons? 
        Of course, we have to take care that indices are correctly mapped...

        Flags: ['I', 'F_*'] -> if F_*, structure will be ignored
        
        The method takes displaced elements and their correcponding threshold in form of: {'element': tuple(displacement_threshold_min,displacement_threshold_max)
        """

        structures_type = f'{structure_type}_structures'

        if not hasattr(self, 'concatenated_map'):
            self.map_stransitions(structure_type=structure_type)

        if not hasattr(self.concatenated_structures, 'structures_grouped_by_composition'):
            self.concatenated_structures.group_by_composition(specorder=specorder) 
        
        if any(not value for value in self.stransitions_properties.values()):
            self.get_all_properties()
        
        # Initialize the `prefilter_dictionary`
        self.prefilter_dictionary = {composition: {'CR_similarity': ['I'] * len(self.concatenated_structures.structures_grouped_by_composition[composition]['indices'])} for composition in self.concatenated_structures.structures_grouped_by_composition.keys()}

        if mode is None:
            pass
        elif mode == 'displacement':            

            for key in self.stransitions_properties.keys():

                transition_displacements = self.stransitions_properties[key]['transition_maximum_displacement_all_structures']
                #
                for element, threshold in displaced_elements.items():
                    structure_index = transition_displacements[element]['structure_index']
                    maximum_displacement = transition_displacements[element]['maximum_displacement']
                    
                    index_in_cm = self.concatenated_map[key][structure_index]
                    structure = self.concatenated_structures.structures[index_in_cm]
                    
                    # Make sure the compostion is in specorder
                    sorted_composition = sorted(structure.get_chemical_symbols(), key=lambda item: specorder.index(item))
                    composition_dict = camus.utils.create_index_dict(sorted_composition)
                    composition_list = [0]*len(specorder)

                    for i, spec in enumerate(composition_dict.keys()):
                        composition_list[i] = len(composition_dict[spec])
                    structure_composition = tuple(composition_list)
                    index_in_gbc = self.concatenated_structures.structures_grouped_by_composition[structure_composition]['indices'].index(index_in_cm)
                    
                    if threshold[0] <= maximum_displacement <= threshold[1]:
                        self.prefilter_dictionary[structure_composition]['CR_similarity'][index_in_gbc] = 'U_PRF'
                    else:
                        self.prefilter_dictionary[structure_composition]['CR_similarity'][index_in_gbc] = 'N_PRF'
                            
        elif mode == 'activation_energy':

            for key in self.stransitions_properties.keys():
                activation_energy = self.stransitions_properties[key]['activation_e_forward']
                structure_index = np.array(self.stransitions_properties[key]['all_energies']).argmax()
                if activation_energy > activation_energy_threshold:
                    self.prefilter_dictionary[key][structure_index] = 'U_PRF'
                else:
                    self.prefilter_dictionary[key][structure_index] = 'N_PRF'

    
    def compare_to_reference(self, reference_set, structure_type='transition', specorder=None, similarity_threshold=0.90,
            metric='laplacian', gamma=1, **ascf_kwargs):
        """
        Creates self._CR_dictionary_composition for self.concatenated_structures and a reference set.
        Also transforms the dictionary to self._CR_dictionary_stransitions of the form {calculation_label: data}.

        `reference_set` must be a Structures object.
        """

        structures_type = f'{structure_type}_structures'

        if not hasattr(self, 'concatenated_map'):
            self.map_stransitions(structure_type=structure_type)

        self._CR_dictionary_composition = compare_sets(reference_set=reference_set, candidate_set=self.concatenated_structures, specorder=specorder, 
                similarity_threshold=similarity_threshold, metric=metric, gamma=gamma, **ascf_kwargs)

        # Transform the self._CR_dictionary_composition dictionary to {calc_label: data} format

        self._CR_dictionary_stransitions = {calculation_label: {
            'CR_similarity': ['I'] * len(self.concatenated_map[calculation_label]),
            'maximum_similarity': [0.0] * len(self.concatenated_map[calculation_label]),
            'max_sim_R_index': [-1] * len(self.concatenated_map[calculation_label])
            } 
                for calculation_label in self.concatenated_map.keys()} 

        for composition in self._CR_dictionary_composition:

            for i, index in enumerate(self.concatenated_structures.structures_grouped_by_composition[composition]['indices']):

                calculation_label = self.inverse_map[index]['calculation_label']
                original_index = self.inverse_map[index]['original_index']

                CR_similarity = self._CR_dictionary_composition[composition]['CR_similarity'][i]
                maximum_similarity = self._CR_dictionary_composition[composition]['maximum_similarity'][i]
                max_sim_R_index= self._CR_dictionary_composition[composition]['max_sim_R_index'][i]

                self._CR_dictionary_stransitions[calculation_label]['CR_similarity'][original_index] = CR_similarity
                self._CR_dictionary_stransitions[calculation_label]['maximum_similarity'][original_index] = maximum_similarity
                self._CR_dictionary_stransitions[calculation_label]['max_sim_R_index'][original_index] = max_sim_R_index


    def cluster_stransitions(self, specorder=None, structure_type='transition', additional_flags_dictionary=None, similarity_threshold=0.90,
                metric='laplacian', gamma=1, **acsf_kwargs):
        """
        Uses cluster_set() to cluster STransitions structures and creates a self._cluster_dictionary_stransitions of form
        'cluster_centers': [{calculation_label: ..., index_in_STransition: ...}],
        'cluster_neighbors': [{calculation_label: ..., index_in_STransition: ...}],
        'cluster_neighbors_similarities': [similarities],
        'orphans': [{calculation_label: ..., index_in_STransition: ...}],
        'prefiltered': [{calculation_label: ..., index_in_STransition: ...}],
        'similarity_flags': ['I'] * len(candidate_set.structures_grouped_by_composition[composition]['indices'])
        """

        structures_type = f'{structure_type}_structures'

        if not hasattr(self, 'concatenated_map'):
            self.map_stransitions(structure_type=structure_type)

        self._cluster_dictionary_compositions = cluster_set(candidate_set=self.concatenated_structures, specorder=specorder, 
                additional_flags_dictionary=additional_flags_dictionary, similarity_threshold=similarity_threshold,
                metric=metric, gamma=gamma, **acsf_kwargs)

        compositions = self._cluster_dictionary_compositions.keys()

        all_cluster_centers = []
        all_cluster_neighborlists = []
        all_similarities = []
        all_orphans = []
        all_prefiltered = []
        all_similarity_result_flags = []

        for composition in compositions:

            for cluster_center in self._cluster_dictionary_compositions[composition]['cluster_centers_indices']:
                calculation_label, index_in_stransition = self.inverse_map[cluster_center].values()
                all_cluster_centers.append({'calculation_label': calculation_label, 'index': index_in_stransition})

            for i, cluster_neighbors in enumerate(self._cluster_dictionary_compositions[composition]['cluster_neighbors_indices']):

                current_neighborlist = []

                for cluster_neighbor in cluster_neighbors:
                    calculation_label, index_in_stransition = self.inverse_map[cluster_neighbor].values()
                    current_neighborlist.append({'calculation_label': calculation_label, 'index': index_in_stransition})

                all_cluster_neighborlists.append(current_neighborlist)
                all_similarities.append(self._cluster_dictionary_compositions[composition]['cluster_neighbors_similarities'][i])

            for orphan in self._cluster_dictionary_compositions[composition]['orphans_indices']:
                calculation_label, index_in_stransition = self.inverse_map[orphan].values()
                all_orphans.append({'calculation_label': calculation_label, 'index': index_in_stransition})

            for prefiltered in self._cluster_dictionary_compositions[composition]['prefiltered_indices']:
                calculation_label, index_in_stransition = self.inverse_map[prefiltered].values()
                all_prefiltered.append({'calculation_label': calculation_label, 'index': index_in_stransition})
            for i, index in enumerate(self.concatenated_structures.structures_grouped_by_composition[composition]['indices']):
                calculation_label, index_in_stransition = self.inverse_map[index].values()
                similarity_flag = self._cluster_dictionary_compositions[composition]['similarity_result_flags'][i]
                all_similarity_result_flags.append({'calculation_label': calculation_label, 'index': index_in_stransition, 'similarity_flag': similarity_flag})


        self._cluster_dictionary_stransitions = {
                'cluster_centers': all_cluster_centers, 
                'cluster_neighbors': all_cluster_neighborlists,
                'cluster_neighbors_similarities': all_similarities,
                'orphans': all_orphans,
                'prefiltered': all_prefiltered,
                'similarity_flags': all_similarity_result_flags
                }


    def create_uniqueness_dictionary(self):
        """
        Creates self._uniqueness_dictionary for future evaluation against reference based on self._cluster_dictionary_stransitions.
        """

        # Creates `self._uniqueness_dictionary` of form {calculation_label: flags},
        # where `flags` is a list ['U_NBR', 'U_CTR', ... , 'N_D'] of length = # of transition structures

        self._uniqueness_dictionary = camus.utils.new_dict()
        self._stopping_index = camus.utils.new_dict()

        for calculation_label in self.concatenated_map.keys():
            
            no_of_structures = len(self.concatenated_map[calculation_label])
            initialize_list = ['I'] * no_of_structures

            self._uniqueness_dictionary[calculation_label] = initialize_list
            self._stopping_index[calculation_label] = -1

        for orphan in self._cluster_dictionary_stransitions['orphans']:
            calculation_label, index = orphan.values()
            self._uniqueness_dictionary[calculation_label][index] = 'U_ORP'

        for i, cluster_center in enumerate(self._cluster_dictionary_stransitions['cluster_centers']):
            calculation_label, index = cluster_center.values()
            self._uniqueness_dictionary[calculation_label][index] = 'U_CTR'

            for cluster_neighbor in self._cluster_dictionary_stransitions['cluster_neighbors'][i]:
                calculation_label_n, index_n = cluster_neighbor.values()
                self._uniqueness_dictionary[calculation_label_n][index_n] = 'U_NBR'

        for prefiltered in self._cluster_dictionary_stransitions['prefiltered']:
            calculation_label, index = prefiltered.values()

            similarity_flag_entry = next(item for item in self._cluster_dictionary_stransitions['similarity_flags'] 
                    if (item['calculation_label'] == calculation_label and item['index'] == index))

            similarity_flag = similarity_flag_entry['similarity_flag']
            self._uniqueness_dictionary[calculation_label][index] = similarity_flag


    def create_evaluation_dictionary(self):
        """
        Creates self._evaluation_dictionary for future evaluation against reference based on self._cluster_dictionary_stransitions and self._uniqueness_dictionary.
        """

        if not hasattr(self, '_uniqueness_dictionary'):
            self.create_uniqueness_dictionary()

        self._evaluation_dictionary = camus.utils.new_dict()

        for calculation_label in self.concatenated_map.keys():

            no_of_structures = len(self.concatenated_map[calculation_label])
            initial_status_list = ['I'] * no_of_structures

            self._evaluation_dictionary[calculation_label] = {
                    'status_list': initial_status_list,
                    'waiting_for': [None] * no_of_structures,
                    'neighbor_of': [None] * no_of_structures
                    }


    def get_maximum_displacement_properties(self, structure_type='minima'):

        """
        structure_type = 'minima' or 'saddlepoints' or 'transition'
        """

        if not any(self.stransitions_properties.values()):
            self.get_all_properties()

        maximum_displacement_all_structures_properties = [self.stransitions_properties[key][f'{structure_type}_maximum_displacement_all_structures'] for key in self.stransitions.keys()]
        
        average_displacement_at_maximum = [self.stransitions_properties[key][f'{structure_type}_average_displacement_per_type'] for key in self.stransitions.keys()]

        chemical_species = self.stransitions_properties[list(self.stransitions.keys())[0]][f'{structure_type}_maximum_displacement_all_structures'].keys() 
       
        return maximum_displacement_all_structures_properties, average_displacement_at_maximum, chemical_species


    def plot_maximum_displacement(self, structure_type='minima', displacement_difference=False, savefig=False, dpi=450, fname = 'max_displacement', save_format='pdf', return_fig_ax=False):

        maximum_displacement_all_structures_properties, average_displacement_at_maximum, chemical_species = self.get_maximum_displacement_properties(structure_type=structure_type)

        no_of_properties = len(maximum_displacement_all_structures_properties)
        property_indices = np.arange(0, no_of_properties, 1)
   
        fig, ax = camus.utils.create_plot(xlabel='', ylabel=r'Displacement $(\AA)$', fontsize=18)

        total_width = 0.75
        width = total_width / no_of_properties

        for i, species in enumerate(chemical_species):

            for j, index in enumerate(property_indices):

                x = i + 1 + total_width * (j / no_of_properties - 0.5)

                color1 = 'royalblue'
                color2 = 'salmon'
                color3 = 'aquamarine'

                maximum_displacement = maximum_displacement_all_structures_properties[index][species]
                average_displacement = average_displacement_at_maximum[index][species]

                if not displacement_difference:
                    bar_max = ax.bar(x, maximum_displacement['maximum_displacement'], width=width, color=color1, align='edge', edgecolor='black', linewidth=0.7, label='Maximum displacement')
                    bar_avg = ax.bar(x, average_displacement[maximum_displacement['structure_index']], width=width, color=color2, align='edge', edgecolor='black', linewidth=0.7, label='Average displacement')
                    
                    #Legend
                    ax.legend(handles=[bar_max, bar_avg], loc='best', fontsize=15)

                else:
                    bar_diff = ax.bar(x, abs(maximum_displacement['maximum_displacement']-average_displacement[maximum_displacement['structure_index']]), width=width, color=color3, align='edge', edgecolor='black', linewidth=0.7, label='Displacement difference')
 
                    #Legend
                    ax.legend(handles=[bar_diff], loc='best', fontsize=15)

        xticks = list(chemical_species)
        plt.xticks(np.arange(1, len(xticks) + 1), xticks, fontsize=18)
        ax.minorticks_on()
        ax.tick_params(axis='y', direction='in', which='both', labelsize=15, length=8)
        ax.tick_params(axis='y', which='minor', length=4)
        ax.tick_params(axis='x', which='both', bottom=False)

        
        # Save plot

        if savefig == False:
            plt.show()
        elif savefig == True:
            plt.savefig(fname=fname + '.' + save_format, format=save_format, bbox_inches='tight', dpi=dpi)

        if return_fig_ax:
            return fig, ax


    def plot_energy_v_displacement(self, chemical_species = None, savefig=False, dpi=450, fname = 'energy_v_displacement', save_format='pdf', return_fig_ax=False):

        if chemical_species is not None:
            chemical_species = chemical_species
            maximum_displacement_all_structures_properties = self.get_maximum_displacement_properties(structure_type='saddlepoints')[0]
        else:
            maximum_displacement_all_structures_properties, _, chemical_species = self.get_maximum_displacement_properties(structure_type='saddlepoints')

        species_colors = ['midnightblue', 'firebrick', 'limegreen', 'gold']
        
        fig, ax = camus.utils.create_plot(xlabel=r'Displacement $(\AA)$', ylabel='Energy (eV)', fontsize=18)

        for i, species in enumerate(chemical_species):
            
            for index, key in enumerate(self.stransitions.keys()):
        
                top_energy = np.max(self.stransitions_properties[key]['all_energies'])
                top_index = int(np.where(self.stransitions_properties[key]['all_energies'] == top_energy)[0])

                maximum_displacement = maximum_displacement_all_structures_properties[index][species]
                structure_index = maximum_displacement['structure_index']

                if structure_index < top_index:
                    energy = self.stransitions_properties[key]['all_energies'][structure_index] - self.stransitions_properties[key]['all_energies'][0]
                else:
                    energy = self.stransitions_properties[key]['activation_e_forward']
                scatter = ax.scatter(energy, maximum_displacement['maximum_displacement'], color=species_colors[i], edgecolor='black', linewidth=0.7, alpha=0.8, s=75, label=f'{species}')

        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markeredgecolor='black', markersize=10, label=label) for color, label in zip(species_colors, chemical_species)]
        ax.legend(handles=legend_handles, labels=list(chemical_species), loc='best', fontsize=15)

        # Save plot

        if savefig == False:
            plt.show()
        elif savefig == True:
            plt.savefig(fname=fname + '.' + save_format, format=save_format, bbox_inches='tight', dpi=dpi)

        if return_fig_ax:
            return fig, ax


    def sisyphus2dft(self, reference_set, base_directory=None, mode=None, specorder=None, structure_type='transition', additional_flags_dictionary=None,
        similarity_threshold=0.90, metric='laplacian', gamma=1, save_traj=True, traj_filename='dft_input.traj', displaced_elements=None, activation_energy_threshold=None, **acsf_kwargs):

        """
        - `reference_set` should be the set used in the training of the currectly used potential
        - `base_directory` will be the same for the coming DFT calculations
        - `mode`:   None ... does basic clustering 
                    `displacement` ... prefilters the structures based on displacement input `displaced_elements` as a dictionary containing {element: (min,max)}
                    `trainsition_energy` ... prefilters the strainsitions base on their `activation_energy`           
        """
        # This should be the same `base_directory` that will then get used in the DFT calculation
        if base_directory is not None:
            base_directory = base_directory
        else:
            base_directory = os.path.join(os.getcwd(), 'base_dft')
        
        # Create base directory if it does not exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        if any(not value for value in self.stransitions_properties.values()):
            self.get_all_properties()

        # The default is to cluster the set without any extra conditions
        if mode is None:

            # Cluster if not previously clustered
            if not hasattr(self, '_cluster_dictionary_stransitions'):
                self.cluster_stransitions(specorder=specorder, structure_type=structure_type, similarity_threshold=similarity_threshold, 
                        metric=metric, gamma=gamma, **acsf_kwargs)
                

        elif mode=='displacement':

            # Prefilter if not previously done
            if additional_flags_dictionary is None:
                self.prefilter_stransitions(mode=mode, specorder=specorder, displaced_elements=displaced_elements)
                additional_flags_dictionary = self.prefilter_dictionary

            # Cluster, taking into acount the prefiltered structures in the `additional_flags_dictionary` 
            if not hasattr(self, '_cluster_dictionary_stransitions'):
                self.cluster_stransitions(specorder=specorder, structure_type=structure_type, additional_flags_dictionary=additional_flags_dictionary, 
                        similarity_threshold=similarity_threshold, metric=metric, gamma=gamma, **acsf_kwargs)

        elif mode=='transition_energy':
            pass

        # Pick only the representative structures
        cluster_centers = self._cluster_dictionary_stransitions['cluster_centers']
        orphans = self._cluster_dictionary_stransitions['orphans']
        if additional_flags_dictionary is not None:
            prefiltered = self._cluster_dictionary_stransitions['prefiltered']
        else:
            prefiltered = []

        # Create a single list of all `candidate` structures
        candidates = cluster_centers + orphans + prefiltered

        # `compare_to_reference`
        self.compare_to_reference(reference_set=reference_set, specorder=specorder, similarity_threshold=similarity_threshold, 
                metric=metric, gamma=gamma, **acsf_kwargs)
        
        # Initialize
        eval_dictionary = {}
        input_structures = []
        
        for candiate in candidates:
            calculation_label = candiate['calculation_label']
            index = candiate['index']

            if self._CR_dictionary_stransitions[calculation_label]['CR_similarity'][index].startswith('U_'):

                eval_label = f'{calculation_label}-{index}'
                eval_dictionary[f'{eval_label}'] = {
                    'origin':{
                        'stransition_label': calculation_label,
                        'stransition_index': index,
                        'sisyphus_dictionary': self._sisyphus_dictionary_path,
                        },
                    'dft_directory': None,
                    'ml_energy': self.stransitions_properties[calculation_label]['all_energies'][index],
                    'ml_forces': [],
                    'dft_energy': None,
                    'dft_forces': [],
                    'dft_flag': 'X',
                    'evaluation_flag': 'U', #`U`...(U)ndecided, `R`...for (R)etraining, `D`...(D)iscard
                    'composition': camus.utils.create_index_dict(self.stransitions[calculation_label].transition_structures.structures[index].get_chemical_symbols())}

                # Create the input set that will go into create VASP
                structure = self.stransitions[calculation_label].transition_structures.structures[index]
                input_structures.append(structure)
 
        # Save the dictionary
        camus.utils.save_to_pickle(eval_dictionary, os.path.join(base_directory, 'eval_dictionary.pkl'))

        # Save a `.traj` file for a simple transition into the DFT stage
        if save_traj:
            write(os.path.join(base_directory, traj_filename), input_structures)


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


def compare_sets(reference_set, candidate_set, specorder=None, similarity_threshold=0.90, 
        metric='laplacian', gamma=1, **acsf_kwargs):
    """
    Generates a dictionary containing info about the 
    unique set of Structures from `candidate_set` w.r.t. `reference_set`.
    `reference_set_structures` and `candidate_set_structures` must be instances of Structures class.

    The algorithm proceeds as follows:
    (1) If `*_set.descriptors==None`, calculate descriptors using `**acsf_kwargs`
    (2) Divide `*_set* into groups by composition
    (3) Find `reference_set` structures eliminating structures with a similarity > `threshold` w.r.t. `reference_set`
    """

    # (1) Get/calculate descriptors

    if not candidate_set.descriptors:
        candidate_set.set_acsf_parameters(**acsf_kwargs)
        candidate_set.calculate_descriptors()

    if not reference_set.descriptors:
        reference_set.set_acsf_parameters(**acsf_kwargs)
        reference_set.calculate_descriptors()

    # (2) Divide sets into groups by composition

    if not hasattr(candidate_set, 'structures_grouped_by_composition'):
        candidate_set.group_by_composition(specorder=specorder)
    if not hasattr(reference_set, 'structures_grouped_by_composition'):
        reference_set.group_by_composition(specorder=specorder)

    # initialize candidate vs. reference uniqueness report dictionary: 
    # {(composition): {status_flags: [], maximum_similarity: [], index_of_max_similarity_reference_structure: []}}
    # possible status_flags = 'I': initialized, 'U_C': unique by composition, 'U_D': unique by descriptor comparison , 'N_D': not unique by descriptor comparison

    CR_dictionary = {composition: {
        'CR_similarity': ['I'] * len(candidate_set.structures_grouped_by_composition[composition]['indices']),
        'maximum_similarity': [0.0] * len(candidate_set.structures_grouped_by_composition[composition]['indices']),
        'max_sim_R_index': [-1] * len(candidate_set.structures_grouped_by_composition[composition]['indices'])
        } for composition in candidate_set.structures_grouped_by_composition.keys()}

    # set descriptors to groups
    
    for composition in candidate_set.structures_grouped_by_composition.keys():
#        if not candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors:
        indices = candidate_set.structures_grouped_by_composition[composition]['indices']
        candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors = [candidate_set.descriptors[i] for i in indices]

    for composition in reference_set.structures_grouped_by_composition.keys():
#        if not reference_set.structures_grouped_by_composition[composition]['structures'].descriptors:
        indices = reference_set.structures_grouped_by_composition[composition]['indices']
        reference_set.structures_grouped_by_composition[composition]['structures'].descriptors = [reference_set.descriptors[i] for i in indices]

    # get common compositions

    candidate_compositions, reference_compositions = candidate_set.structures_grouped_by_composition.keys(), reference_set.structures_grouped_by_composition.keys()
    common_compositions = candidate_compositions & reference_compositions

    # (3) Iterate over common compositions and calculate similarity between all candidate structures w.r.t. all reference structures

    # if candidate composition not in reference, automatically set unique CR flags ('U_C' -> unique by composition)

    for composition in candidate_compositions:
        if composition not in reference_compositions:
            CR_dictionary[composition]['CR_similarity'] = ['U_C'] * len(CR_dictionary[composition]['CR_similarity']) 

    # iterate over common compositions, calculate similarity

    average_kernel = AverageKernel(metric=metric, gamma=gamma) 

    for composition in common_compositions:

        candidate_descriptors = candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors
        reference_descriptors = reference_set.structures_grouped_by_composition[composition]['structures'].descriptors

        similarity_kernel = average_kernel.create(candidate_descriptors, reference_descriptors)

        for candidate_index, similarity_list in enumerate(similarity_kernel):
            
            maximum_similarity = np.max(similarity_list)
            max_sim_reference_index_in_kernel = np.argmax(similarity_list)

            max_sim_reference_index = reference_set.structures_grouped_by_composition[composition]['indices'][max_sim_reference_index_in_kernel] 

            if maximum_similarity > similarity_threshold:
                CR_flag = 'N_D'
            else:
                CR_flag = 'U_D'

            CR_dictionary[composition]['CR_similarity'][candidate_index] = CR_flag
            CR_dictionary[composition]['maximum_similarity'][candidate_index] = maximum_similarity
            CR_dictionary[composition]['max_sim_R_index'][candidate_index] = max_sim_reference_index

    return CR_dictionary


def cluster_set(candidate_set, specorder=None, additional_flags_dictionary=None, similarity_threshold=0.90, 
        metric='laplacian', gamma=1, n_jobs=1, positions=None, **acsf_kwargs):
    """
    Cluster `candidate_set` around structures with the maximum number of similar neighbors.
    Returns `cluster_dictionary` listing the centers of clusters, etc... 
    If `additional_flags_dictionary` is given, such as a `CR_dictionary`, will ignore structures with 'N_*' flags. Must be shaped as `CR_dictionary`.
    """

    # (1) Get/calculate descriptors

    if not candidate_set.descriptors:
        candidate_set.set_acsf_parameters(**acsf_kwargs)
        candidate_set.calculate_descriptors(n_jobs=n_jobs, positions=positions)

    # (2) Divide sets into groups by composition

    if not hasattr(candidate_set, 'structures_grouped_by_composition'):
        candidate_set.group_by_composition(specorder=specorder)
 
    # set descriptors to groups

    for composition in candidate_set.structures_grouped_by_composition.keys():
#        if not candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors:
        indices = candidate_set.structures_grouped_by_composition[composition]['indices']
        candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors = [candidate_set.descriptors[i] for i in indices]

    # initialize cluster_dictionary

    cluster_dictionary = {composition: {
        'cluster_centers_indices': camus.utils.new_list(),
        'cluster_neighbors_indices': camus.utils.new_list(),
        'cluster_neighbors_similarities': camus.utils.new_list(),
        'orphans_indices': camus.utils.new_list(),
        'prefiltered_indices': camus.utils.new_list(),
        'similarity_result_flags': ['I'] * len(candidate_set.structures_grouped_by_composition[composition]['indices'])
        } for composition in candidate_set.structures_grouped_by_composition.keys()}

    # (3) Calculate similarities between all structures

    average_kernel = AverageKernel(metric=metric, gamma=gamma)

    for composition in candidate_set.structures_grouped_by_composition.keys():

        # get True, False list for indices not flagged with 'N_*'    
        if additional_flags_dictionary:
            additional_flags = additional_flags_dictionary[composition]['CR_similarity']
            flags_boolean = [True if not flag.startswith('N_') else False for flag in additional_flags]

        else:
            flags_boolean = [True] * len(candidate_set.structures_grouped_by_composition[composition]['indices'])
        
        # indices of structures to be clustered
        relevant_indices = [index for index, value in enumerate(flags_boolean) if value]
        
        # indices of structures flagged with 'N_*'
        prefiltered_indices = [index for index, value in enumerate(flags_boolean) if not value]

        # save original structure indices
        original_structure_indices = [candidate_set.structures_grouped_by_composition[composition]['indices'][i]
                for i in relevant_indices]

        # get descriptors of structures to be clustered
        relevant_descriptors = [candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors[i] 
                for i in relevant_indices]

        # save original prefiltered structures indices (those flagged with 'N_*')
        prefiltered_structure_indices = [candidate_set.structures_grouped_by_composition[composition]['indices'][i]
                for i in prefiltered_indices]

        cluster_dictionary[composition]['prefiltered_indices'] = prefiltered_structure_indices

        # flag prefiltered structures
        for i in prefiltered_indices:
            cluster_dictionary[composition]['similarity_result_flags'][i] = additional_flags[i]

        # calculate similarity kernel
        similarity_kernel = average_kernel.create(relevant_descriptors)

        # neighbor counting loop
        
        orphans_structure_indices = original_structure_indices.copy()
        mask = similarity_kernel > similarity_threshold
        np.fill_diagonal(mask, False)

        while not np.all(mask == False):

            neighbor_counters = np.sum(mask, axis=1) 

            maximum_counter_index = np.argmax(neighbor_counters)

            # save center original index
            center_original_index = original_structure_indices[maximum_counter_index]

            # save neighbors indices and similarities
            mask_true_indices = [index for index, value in enumerate(mask[maximum_counter_index]) if value]

            neighbors_original_indices = [original_structure_indices[i] for i in mask_true_indices]
            neighbors_similarities = similarity_kernel[maximum_counter_index][mask[maximum_counter_index]]

            cluster_dictionary[composition]['cluster_centers_indices'].append(center_original_index)
            cluster_dictionary[composition]['cluster_neighbors_indices'].append(neighbors_original_indices)
            cluster_dictionary[composition]['cluster_neighbors_similarities'].append(neighbors_similarities)
            cluster_dictionary[composition]['similarity_result_flags'][relevant_indices[maximum_counter_index]] = 'U_CTR'

            for i in mask_true_indices:
                cluster_dictionary[composition]['similarity_result_flags'][relevant_indices[i]] = 'U_NBR'

            for i in range(len(mask)):

                mask[i][maximum_counter_index] = False

                for j in mask_true_indices:
                    mask[i][j] = False

            orphans_structure_indices = [index for index in orphans_structure_indices 
                    if ((index not in neighbors_original_indices) and index != center_original_index)]


        # save and flag orphans
        cluster_dictionary[composition]['orphans_indices'] = orphans_structure_indices

        orphans_indices = [relevant_indices[index] for index, value in enumerate(original_structure_indices) if value in set(orphans_structure_indices)]

        for i in orphans_indices:
            cluster_dictionary[composition]['similarity_result_flags'][i] = 'U_ORP'

    return cluster_dictionary


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
        composition = camus.utils.create_index_dict(input_structure.get_chemical_symbols())
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

    eval_dictionary = camus.utils.load_pickle(eval_dictionary_path)

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
    camus.utils.save_to_pickle(eval_dictionary, os.path.join(base_directory, 'eval_dictionary.pkl'))


def plot_evaluation(path_to_eval_dictionary, xlabel=r'$E_{ML} - E_{self}$', ylabel=r'$E_{DFT} - E_{self}$', sizex=15.0, sizey=15.0, fontsize=15, fit=False, histogram=False):
    """
    `fit` might make sense to show some sort of progress over several runs
    `histogram` to make the assesment of the correct energy interval for retraing easier
    """

    eval_dictionary = camus.utils.load_pickle(path_to_eval_dictionary)
    ml_energies = []
    dft_energies = []
    for key in eval_dictionary.keys():
        ml_energies.append(eval_dictionary[key]['corrected_ml_energy'])
        dft_energies.append(eval_dictionary[key]['corrected_dft_energy'])

    if not histogram:
        fig, ax = camus.utils.create_plot(xlabel=xlabel, ylabel=ylabel, sizex=sizex, sizey=sizey, fontsize=fontsize)
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

        fig, ax = camus.utils.create_plot(xlabel='', ylabel=r'Energy difference [eV]', fontsize=18)
        bar_diff = ax.bar(x, energy_diffs, width=width, color='royalblue', align='edge', edgecolor='black', linewidth=0.7, label="|E_DFT-E_ML|")

        ax.tick_params(axis='y', direction='in', which='both', labelsize=15, length=8)
        ax.tick_params(axis='y', which='minor', length=4)

        ax.legend(handles=[bar_diff], loc='best', fontsize=15)
        plt.show()

