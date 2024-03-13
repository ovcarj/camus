"""

This module compares sets and creates clusters of similar structures based on
the descriptors calculated by calculate_descriptors() in the Structures class.

"""

import numpy as np
import camus.tools.utils

from collections import Counter

from dscribe.kernels import AverageKernel


def compare_sets(reference_set, candidate_set, specorder=None, similarity_threshold=0.90, 
        metric='laplacian', gamma=1, **acsf_kwargs):
    """
    Generates a dictionary containing info about the 
    unique set of Structures from `candidate_set` w.r.t. `reference_set`.
    `reference_set_structures` and `candidate_set_structures` must be instances of Structures class.

    The algorithm proceeds as follows:
    (1) If `*_set.descriptors==np.empty(0)` (not yet calculated), calculate descriptors using `**acsf_kwargs`
    (2) Divide `*_set* into groups by composition
    (3) Calculate the `CR_dictionary` for the `candidate_set` using the `insert_reference` algorithm
        Possible status_flags = 'I': initialized, 'U_C': unique by composition, 'U_D': unique by descriptor comparison , 'N_D': not unique by descriptor comparison
    """

    # (1) Get/calculate descriptors

#    if not candidate_set.descriptors.any():
    if candidate_set.descriptors == []:
        candidate_set.set_acsf_parameters(**acsf_kwargs)
        candidate_set.calculate_descriptors()

#    if not reference_set.descriptors.any():
    if reference_set.descriptors == []:
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
        'max_sim_R_index': [-1] * len(candidate_set.structures_grouped_by_composition[composition]['indices']),
        'delta_e_CR': [0.0] * len(candidate_set.structures_grouped_by_composition[composition]['indices'])
        } for composition in candidate_set.structures_grouped_by_composition.keys()}

    # get reference_set energies for later writing (forces will possibly be added later, for now not necessary)
    reference_energies = reference_set.get_energies_and_forces()[0]

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

        candidate_energies = candidate_set.structures_grouped_by_composition[composition]['structures'].get_energies_and_forces()[0]

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
            CR_dictionary[composition]['delta_e_CR'][candidate_index] = candidate_energies[candidate_index] - reference_energies[max_sim_reference_index]

    return CR_dictionary


def cluster_set(candidate_set, specorder=None, additional_flags_dictionary=None, similarity_threshold=0.90, 
        metric='laplacian', gamma=1, n_jobs=1, positions=None, **acsf_kwargs):
    """
    Cluster `candidate_set` around structures with the maximum number of similar neighbors.
    Returns `cluster_dictionary` listing the centers of clusters, etc... 
    If `additional_flags_dictionary` is given, such as a `CR_dictionary`, will ignore structures with 'N_*' flags. Must be shaped as `CR_dictionary`.
    """

    # (1) Get/calculate descriptors

#    if not candidate_set.descriptors.any():
    if candidate_set.descriptors == []:
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
        'cluster_centers_indices': camus.tools.utils.new_list(),
        'cluster_neighbors_indices': camus.tools.utils.new_list(),
        'cluster_neighbors_similarities': camus.tools.utils.new_list(),
        'orphans_indices': camus.tools.utils.new_list(),
        'prefiltered_indices': camus.tools.utils.new_list(),
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


def find_cluster_center(cluster_dictionary_stransitions, target_calculation_label, target_index):
    """ 
    Intended to return the cluster center of the corresponding 
    (calculation_label, neighbor_index) pair
    """
    for index, sublist in enumerate(cluster_dictionary_stransitions['cluster_neighbors']):
        for item in sublist:
            if item['calculation_label'] == target_calculation_label and item['index'] == target_index:
                return cluster_dictionary_stransitions['cluster_centers'][index]

