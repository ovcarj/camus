import importlib
import click

class CalcMap():
    """
    Maps from calculation types, methods and engines to the corresponding Calc classes. 

    TODO: a better implementation of this mapping could be thought of so there is no hardcoding.
    Perhaps automatically searching the codebase for (calc_type, method, engine) class attributes...

    """

    def __init__(self, calc_type=None, method=None, engine=None):
        """
        List all implemented calculation types, methods and engines. If given, map the calculation type, method and engine to the corresponding implementation of the Calc class.

        Parameters
        ----------
        calc_type : str | None
            Name of the calculation type
        method : str | None
            Optional calculation method
        engine : str | None
            Optional calculation engine

        """

        if not method:
            method = 'no_method'

        if not engine:
            engine = 'no_engine'

        self.calc_type = calc_type
        self.method = method
        self.engine = engine

        self._define_maps()
        self._get_full_names_dict()
        self._get_calc_types_dict()

        if calc_type:
            self.calc_class = self._get_calc_class(self.calc_type, self.method, self.engine)

    def _define_maps(self):
        """
        Defines maps from (calc_type, method, engine) to Calc modules and classes.

        """

        self.maps = {

                ('energy_force', 'classical_potential', 'LAMMPS'): 
                    ['camus.calc.LAMMPS.energy_force.lammps_cl_enfor_calc', 'LAMMPSClEnForCalc'],

                # The following are temporarily written just for testing purposes

                ('energy_force', 'neural_network', 'LAMMPS'): 
                    ['camus.calc.LAMMPS.energy_force.lammps_nn_enfor_calc', 'LAMMPSNNEnForCalc'],

                ('geo_rlx', 'classical_potential', 'LAMMPS'): 
                    ['camus.calc.LAMMPS.energy_force.lammps_cl_geo_rlx_calc', 'LAMMPSGeoRlxCalc'],

                ('structure_analysis', 'no_method', 'no_engine'): 
                    ['camus.calc.LAMMPS.energy_force.lammps_cl_geo_rlx_calc', 'LAMMPSGeoRlxCalc'],
                
                }

    def _get_calc_class(self, calc_type, method=None, engine=None):
        """
        Return the implementation of the Calc class corresponding to (calc_type, method, engine).

        Parameters
        ----------
        calc_type : str
            Name of the calculation type
        method : str
            Name of the calculation method
        engine : str
            Name of the calculation engine

        Returns
        ----------
        calc_class : class
            Implementation of the Calc class corresponding to calculation type, method and engine

        """
        
        if not method:
            method = 'no_method'

        if not engine:
            engine = 'no_engine'

        class_module_name, class_name = self.maps.get((calc_type, method, engine))

        class_module = importlib.import_module(class_module_name)
        calc_class = getattr(class_module, class_name)

        return calc_class

    def _get_calc_types_dict(self):
        """
        Creates and a dictionary ``self._all_calctypes = {1: calc_type1, 2: calc_type2, ...}``.

        """

        calc_types = [calc[0] for calc in self.maps.keys()]
        unique_calc_types = sorted(list(set(calc_types)))

        self._all_calc_types = {
                f'{i + 1}': calc_type for i, calc_type in enumerate(unique_calc_types)
                }

    def _get_methods_dict(self, calc_type):
        """
        For a given ``calc_type``, return a dictionary of implemented methods.

        Parameters
        ----------
        calc_type : str
            Name of the calculation type

        Returns
        ----------
        methods : dict
            Dictionary of implemented methods for a given calculation type

        """

        all_methods = [calc[1] for calc in self.maps.keys() if calc[0] == calc_type]
        unique_methods = sorted(list(set(all_methods)))

        methods = {
                f'{i + 1}': method for i, method in enumerate(unique_methods)
                }

        return methods

    def _get_engines_dict(self, calc_type, method=None):
        """
        For a given ``calc_type`` and ``method``, return a dictionary of implemented methods.

        Parameters
        ----------
        calc_type : str
            Name of the calculation type
        method : str | None
            Name of the method

        Returns
        ----------
        engines : dict
            Dictionary of implemented engines for a given calculation type and method

        """

        if not method:
            method = 'no_method'

        all_engines = [calc[2] for calc in self.maps.keys() if (calc[0] == calc_type and calc[1] == method)]
        unique_engines = sorted(list(set(all_engines)))

        engines = {
                f'{i + 1}': engine for i, engine in enumerate(unique_engines)
                }

        return engines

    def _get_full_names_dict(self):
        """
        Creates a dict from keywords to full names, e.g. {'energy_force': 'Energy-force calculation'}.

        """

        self._full_names_dict = {
                'energy_force': 'Single point energy force calculation',
                'geo_rlx': 'Geometry relaxation',
                'structure_analysis': 'Structure analysis',

                'no_method': 'No method',
                'classical_potential': 'Classical potential',
                'dft': 'DFT',
                'nn': 'Neural network',
                'neural_network': 'Neural network',

                'no_engine': 'No engine',
                'lammps': 'LAMMPS',
                'vasp': 'VASP',
                'LAMMPS': 'LAMMPS',
                'VASP': 'VASP'}

    def _pretty_print_dict(self, dictionary):
        """
        Prints f'({key})' self._full_names_dict[{value}]' for keys and values in a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary whose keys and full names of values are being printed

        """

        for key, value in dictionary.items():
            full_name = self._full_names_dict[value]
            click.echo(f'({key}) {full_name}')
