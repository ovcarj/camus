""" Definition of the STransitions class.

This module analyzes a collection of Sisyphus transitions.

"""

import random
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import warnings

import camus.tools.utils
from camus.structures import Structures
from camus.stransition import STransition
from camus.tools.structure_comparison import cluster_set, compare_sets, find_cluster_center


class STransitions():

    def __init__(self, sisyphus_dictionary=None, base_directory=None, sisyphus_dictionary_path=None, 
            save_analysis=False, **kwargs):
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
                self._sisyphus_dictionary = camus.tools.utils.load_pickle(sisyphus_dictionary_path)

            else:
                self._sisyphus_dictionary_path = os.path.join(f'{self._base_directory}', 'sisyphus_dictionary.pkl')

            self._sisyphus_dictionary = camus.tools.utils.load_pickle(self._sisyphus_dictionary_path)
            
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
        Creates the inverse dictionary {index in self.concatenated_stransition: calc_label} for convenience to self.concatenated_map_inverse.
        
        `structure_type` can be ['transition', 'minima', 'saddlepoints'].
        """

        structures_type = f'{structure_type}_structures'
        concatenated_structures = []
        calculation_labels = self.stransitions.keys()

        self.concatenated_map = {calculation_label: camus.tools.utils.new_list for calculation_label in calculation_labels}
        self.concatenated_map_inverse = camus.tools.utils.new_dict()

        structure_counter = 0
        
        for calculation_label in calculation_labels:

            structures = self.stransitions[calculation_label].__getattribute__(structures_type).structures
            no_of_structures = len(structures)

            indices = np.arange(structure_counter, structure_counter + no_of_structures)
            self.concatenated_map[calculation_label] = indices
            structure_counter += no_of_structures

            for original_index, new_index in enumerate(indices):

                self.concatenated_map_inverse[new_index]= {'calculation_label': calculation_label, 'original_index': original_index}

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
                    composition_dict = camus.tools.utils.create_index_dict(sorted_composition)
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

                calculation_label = self.concatenated_map_inverse[index]['calculation_label']
                original_index = self.concatenated_map_inverse[index]['original_index']

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
                calculation_label, index_in_stransition = self.concatenated_map_inverse[cluster_center].values()
                all_cluster_centers.append({'calculation_label': calculation_label, 'index': index_in_stransition})

            for i, cluster_neighbors in enumerate(self._cluster_dictionary_compositions[composition]['cluster_neighbors_indices']):

                current_neighborlist = []

                for cluster_neighbor in cluster_neighbors:
                    calculation_label, index_in_stransition = self.concatenated_map_inverse[cluster_neighbor].values()
                    current_neighborlist.append({'calculation_label': calculation_label, 'index': index_in_stransition})

                all_cluster_neighborlists.append(current_neighborlist)
                all_similarities.append(self._cluster_dictionary_compositions[composition]['cluster_neighbors_similarities'][i])

            for orphan in self._cluster_dictionary_compositions[composition]['orphans_indices']:
                calculation_label, index_in_stransition = self.concatenated_map_inverse[orphan].values()
                all_orphans.append({'calculation_label': calculation_label, 'index': index_in_stransition})

            for prefiltered in self._cluster_dictionary_compositions[composition]['prefiltered_indices']:
                calculation_label, index_in_stransition = self.concatenated_map_inverse[prefiltered].values()
                all_prefiltered.append({'calculation_label': calculation_label, 'index': index_in_stransition})

            for i, index in enumerate(self.concatenated_structures.structures_grouped_by_composition[composition]['indices']):
                calculation_label, index_in_stransition = self.concatenated_map_inverse[index].values()
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

        self._uniqueness_dictionary = camus.tools.utils.new_dict()

        for calculation_label in self.concatenated_map.keys():
            
            no_of_structures = len(self.concatenated_map[calculation_label])
            initialize_list = ['I'] * no_of_structures

            self._uniqueness_dictionary[calculation_label] = initialize_list

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
        Creates self._evaluation_dictionary of format (TODO: Write exact format) for future evaluation against reference based on self._cluster_dictionary_stransitions and self._uniqueness_dictionary.
        Possible status_list flags: `P` (pass/good structure), `W` (waiting for some other structure to be evaluated), `BC` (goes into next batch of calculation), `C` (calculation running), `B` (bad prediction), `T` (throw away)
        Structure of `waiting_for`: `index of structure that's waiting for`: `{'calculation_label', index}`
        """

        if not hasattr(self, '_uniqueness_dictionary'):
            self.create_uniqueness_dictionary()

        self._evaluation_dictionary = camus.tools.utils.new_dict()

        for calculation_label in self.stransitions.keys():

            no_of_structures = len(self.stransitions[calculation_label].transition_structures.structures)

            uniqueness_dictionary = np.array(self._uniqueness_dictionary[calculation_label], dtype='str')
            waiting_list = camus.tools.utils.new_dict()
            status_list = np.where(uniqueness_dictionary == 'N_D', 'P', 'I')

            for i in range(no_of_structures):

                uniqueness_flag = uniqueness_dictionary[i]

                if uniqueness_flag == 'N_D':
                    pass

                elif uniqueness_flag == 'U_NBR':

                    status_list[i] = 'W'

                    # Find the corresponding cluster center

                    cluster_center = find_cluster_center(self._cluster_dictionary_stransitions, calculation_label, i)
                    waiting_list[str(i)] = cluster_center

                    if cluster_center['calculation_label'] == calculation_label:
                        pass
                    else:
                        break

                elif (uniqueness_flag == 'U_CTR' or uniqueness_flag == 'U_ORP'):

                    status_list[i] = 'BC'

                    break

            self._evaluation_dictionary[calculation_label] = {
                    'status_list': status_list,
                    'waiting_for': waiting_list
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
   
        fig, ax = camus.tools.utils.create_plot(xlabel='', ylabel=r'Displacement $(\AA)$', fontsize=18)

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
        
        fig, ax = camus.tools.utils.create_plot(xlabel=r'Displacement $(\AA)$', ylabel='Energy (eV)', fontsize=18)

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
                    'composition': camus.tools.utils.create_index_dict(self.stransitions[calculation_label].transition_structures.structures[index].get_chemical_symbols())}

                # Create the input set that will go into create VASP
                structure = self.stransitions[calculation_label].transition_structures.structures[index]
                input_structures.append(structure)
 
        # Save the dictionary
        camus.tools.utils.save_to_pickle(eval_dictionary, os.path.join(base_directory, 'eval_dictionary.pkl'))

        # Save a `.traj` file for a simple transition into the DFT stage
        if save_traj:
            write(os.path.join(base_directory, traj_filename), input_structures)


