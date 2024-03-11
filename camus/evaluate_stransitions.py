""" Definition of the EST (Evaluate STransitions) class.

This module is to handle the evaluation of the accuracy of an ML model against DFT.

"""

import numpy as np
import os
import warnings

import camus.tools.utils

from camus.batch import Batch
from camus.tools.structure_comparison import find_cluster_center


class EST():

    def __init__(self, evaluation_directory, specorder, 
            reference_set=None, stransitions=None, 
            evaluation_dictionary=None, energy_threshold=0.1,
            scheduler='Slurm', dft_engine='VASP'):
        """
        Initializes a new EST object for evaluation of the accuracy of
        structures generated by Sisyphus (STransitions) against DFT.

        `reference_set` should be a Structures object.
        `stransitions` should be an STransitions object.

        `evaluation_dictionary` (TODO: Write exact format) should work with path to pkl or already loaded pkl (a dict)
        Possible status_list flags: `P` (pass/good structure), `W` (waiting for some other structure to be evaluated), `BC` (goes into next batch of calculation), `C` (calculation running), `B` (bad prediction), `T` (throw away)
        Structure of `waiting_for`: `index of structure that's waiting for`: `{'calculation_label', index}`

        `evaluation_directory` is the directory where the evaluation calculations will run.

        Parameters:
            to_be_written (TODO): TODO.
        """

        if (reference_set is None) and (stransitions is None) and (evaluation_dictionary is None):
            warnings.warn("EST is not expected to be initialized without reference_set, stransitions and/or evaluation_dictionary.")

        self.evaluation_directory = evaluation_directory
        self._specorder = specorder

        self.reference_set = reference_set
        self.stransitions = stransitions

        if evaluation_dictionary is not None:
            if type(evaluation_dictionary) == str:
                self.evaluation_dictionary = camus.tools.utils.load_pickle(evaluation_dictionary)
            elif type(evaluation_dictionary) == dict:
                self.evaluation_dictionary = evaluation_dictionary
            else:
                raise TypeError("evaluation_dictionary should be given as a path to a .pkl file or a dict.")

        else:
            self.evaluation_dictionary = evaluation_dictionary

        self.energy_threshold = energy_threshold
        self._scheduler = scheduler
        self._dft_engine = dft_engine

        # Create evaluation base directory if it does not exist
        if not os.path.exists(self.evaluation_directory):
            os.makedirs(self.evaluation_directory)
                

    def initialize_evaluation(self, prefilter_mode=None, prefilter_args=None):
        """
        Calculates the necessary dictionaries for initial evaluation and 
        writes them to the `self.evaluation_directory`.
        """

        if prefilter_mode or prefilter_args:
            raise NotImplementedError("Prefiltering will be implemented soon...")

        # Get CR, cluster and uniqueness STransition dictionaries

        self.stransitions.compare_to_reference(reference_set=self.reference_set, specorder=self._specorder)
        self.stransitions.cluster_stransitions(specorder=self._specorder, additional_flags_dictionary=self.stransitions._CR_dictionary_composition)
        self.stransitions.create_uniqueness_dictionary()

        self.CR_dictionary = self.stransitions._CR_dictionary_stransitions
        self.cluster_dictionary = self.stransitions._cluster_dictionary_stransitions
        self.uniqueness_dictionary = self.stransitions._uniqueness_dictionary

        # Save CR, cluster and uniqueness STransition dictionaries

        camus.tools.utils.save_to_pickle(self.CR_dictionary, os.path.join(self.evaluation_directory, 'CR.pkl'))
        camus.tools.utils.save_to_pickle(self.cluster_dictionary, os.path.join(self.evaluation_directory, 'clusters.pkl'))
        camus.tools.utils.save_to_pickle(self.uniqueness_dictionary, os.path.join(self.evaluation_directory, 'uniqueness.pkl'))

        # Create initial evaluation dictionary

        self.create_evaluation_dictionary()

        # Save initial evaluation dictionary

        camus.tools.utils.save_to_pickle(self.evaluation_dictionary, os.path.join(self.evaluation_directory, 'evaluation.pkl'))


    def create_evaluation_dictionary(self):
        """
        Creates self.evaluation_dictionary of format (TODO: Write exact format) for future evaluation against reference based on self.cluster_dictionary_stransitions and self._uniqueness_dictionary.
        Possible status_list flags: `G` (pass/good prediction), `W` (waiting for some other structure to be evaluated), `P` (prepare for next batch of calculation), `C` (calculation running), `B` (bad prediction), `T` (throw away)
        Structure of `waiting_for`: `index of structure that's waiting for`: `{'calculation_label', index}`
        Structure of `energies`: `index of structure`:{ML energy, DFT energy}
        """

        self.evaluation_dictionary = camus.tools.utils.new_dict()

        for calculation_label in self.stransitions.stransitions.keys():

            no_of_structures = len(self.stransitions.stransitions[calculation_label].transition_structures.structures)

            uniqueness_dictionary = np.array(self.uniqueness_dictionary[calculation_label], dtype='str')
            waiting_list = camus.tools.utils.new_dict()
            status_list = ['I'] * no_of_structures

            # Energy difference between the model and reference (usually DFT)
            delta_e_MR = [None] * no_of_structures

            for i in range(no_of_structures):

                uniqueness_flag = uniqueness_dictionary[i]

                if uniqueness_flag == 'N_D':
                    # This means there's a similar structure in the training set,
                    # so we immediately make a comparison

                    delta_e_MR[i] = self.CR_dictionary[calculation_label]['delta_e_CR'][i] 

                    if np.abs(delta_e_MR[i]) < self.energy_threshold:
                        status_list[i] = 'G'

                    else:
                        status_list[i] = 'B'
                        status_list[i+1:] = 'T'
                        break

                elif uniqueness_flag == 'U_NBR':

                    status_list[i] = 'W'

                    # Find the corresponding cluster center

                    cluster_center = find_cluster_center(self.cluster_dictionary, calculation_label, i)
                    waiting_list[str(i)] = cluster_center

                    if cluster_center['calculation_label'] == calculation_label:
                        pass
                    else:
                        break

                elif (uniqueness_flag == 'U_CTR' or uniqueness_flag == 'U_ORP'):

                    status_list[i] = 'P'
                    # Immediately write the ML energy, prepare for DFT energy

                    break

            self.evaluation_dictionary[calculation_label] = {
                    'status_list': status_list,
                    'waiting_for': waiting_list,
                    'delta_e_MR': delta_e_MR
                    }

