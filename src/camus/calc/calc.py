import abc
import click

class Calc():
    """
    Base class for defining the calculation type, method, engine and corresponding classes.

    """

    def __init__(self, calc_type, 
            method=None, engine=None,
            creator=None, runner=None, pp=None,
            description=''):
        """
        Defines the name of calculation type and the corresponding method, engine and classes.

        Parameters
        ----------
        calc_type : str
            Name of the calculation type (e.g. energy_force, md, ...)
        method : None | str
            Method used for the calculation (e.g. level of theory: classical potential, dft, ...)
        engine : None | str
            Code used for the calculation (e.g. LAMMPS, VASP, ...)
        creator : None | class
            Corresponding CalcCreator class
        runner : None | class
            Corresponding CalcRunner class
        pp : None | class
            Corresponding CalcPP class
        description : str
            Short description of the calculation type

        """

        self.calc_type = calc_type
        self.method = method
        self.engine = engine
        self.creator = creator
        self.runner = runner
        self.pp = pp
        self.description = description
