""" Definition of the STransition class.

This module is used to analyze a single Sisyphus transition.

"""

import numpy as np
import os

import camus.tools.utils
from camus.structures import Structures

from functools import cached_property


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

            self._sisyphus_dictionary = camus.tools.utils.load_pickle(self._sisyphus_dictionary_path)
            
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


