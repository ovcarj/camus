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

from ase.io import write
from ase.io.lammpsrun import read_lammps_dump

from dscribe.descriptors import ACSF
from dscribe.kernels import AverageKernel

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
            self.descriptors = []

        if acsf_parameters is not None:
            self.acsf_parameters = acsf_parameters
        else:
            self.acsf_parameters = {}
            
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
            'g2_params': [(1, 2), (1, 4)], #[(1, 2), (1, 4), (1, 8),(1,16)],
            'g3_params': [1,2],
            'g4_params': [(1, 2, 1), (1, 4, 1)], #[(1, 4, 4), (1, 4, -1), (1, 8, 4), (1, 8, -1)],
            'species': ['Cs', 'Pb', 'Br', 'I'],
            'periodic': True
            }

        for key in kwargs:
            if key not in default_parameters:
                raise RuntimeError('Unknown keyword: %s' % key)

        # Set self.acsf_parameters
        for key, value in default_parameters.items():
            self.acsf_parameters[key] = kwargs.pop(key, value)

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

        # Special case of single input structure:
        if isinstance(input_structures, Atoms): input_structures = [input_structures]

        for atoms in input_structures:
            descriptor = acsf_descriptor.create(atoms)
            self.descriptors.append(descriptor)



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
            fig.savefig(fname=f"{plot_parameters['plot_title'].lower()}.{plot_parameters['save_as']}", bbox_inches='tight', format=plot_parameters['save_as'])
    
        plt.show()



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

        self.stransitions = {key: STransition(stransition=(key, value)) for key, value in self._sisyphus_dictionary.items()}

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


    def filter_stransitions(self, training_set, additional_criteria=None, **acsf_kwargs):
        """
        Filters structures obtained by a batch of Sisyphus runs to create a new training set.
        Descriptors for the old `training set` will be calculated if `training_set.descriptors==[]`.
        """

        # Initialize training set Structures object to self._training set

        self._training_set = training_set

        # Initialize `self.training_flags` dictionary of form {calculation_label: flags},
        # where `flags` is a list ['I', 'I', ... , 'I'] of length = # of transition structures
        # ('I' stands for initialized)

        self.training_flags = dict()

        for calculation_label in self.stransitions.keys():
            
            no_of_structures = len(self.stransitions[calculation_label].transition_structures.structures)
            initialize_list = ['I'] * no_of_structures

            self.training_flags[calculation_label] = initialize_list

        ##############################################
        # Potential preprocessing (maximum displacement, etc.) 
        # using additional_criteria goes here
        ##############################################

        # Create descriptors for the training set

        if (self._training_set.descriptors == []):
            self._training_set.set_acsf_parameters(**acsf_kwargs)
            self._training_set.calculate_descriptors()



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
        'maximum_similarity': [0] * len(candidate_set.structures_grouped_by_composition[composition]['indices']),
        'max_sim_R_index': [-1] * len(candidate_set.structures_grouped_by_composition[composition]['indices'])
        } for composition in candidate_set.structures_grouped_by_composition.keys()}

    # set descriptors to groups
    
    for composition in candidate_set.structures_grouped_by_composition.keys():
        if not candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors:
            indices = candidate_set.structures_grouped_by_composition[composition]['indices']
            candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors = [candidate_set.descriptors[i] for i in indices]

    for composition in reference_set.structures_grouped_by_composition.keys():
        if not reference_set.structures_grouped_by_composition[composition]['structures'].descriptors:
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
        metric='laplacian', gamma=1, **acsf_kwargs):
    """
    Cluster `candidate_set` around structures with the maximum number of similar neighbors.
    Returns `cluster_dictionary` listing the centers of clusters, etc... 
    If `additional_flags_dictionary` is given, such as a `CR_dictionary`, will ignore structures with 'N_*' flags. Must be shaped as ``
    """

    # (1) Get/calculate descriptors

    if not candidate_set.descriptors:
        candidate_set.set_acsf_parameters(**acsf_kwargs)
        candidate_set.calculate_descriptors()

    # (2) Divide sets into groups by composition

    if not hasattr(candidate_set, 'structures_grouped_by_composition'):
        candidate_set.group_by_composition(specorder=specorder)
 
    # set descriptors to groups

    for composition in candidate_set.structures_grouped_by_composition.keys():
        if not candidate_set.structures_grouped_by_composition[composition]['structures'].descriptors:
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
