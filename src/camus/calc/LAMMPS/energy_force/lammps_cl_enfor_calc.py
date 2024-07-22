from camus.calc.calc import Calc
from camus.calc.LAMMPS.energy_force.lammps_cl_enfor_creator import LAMMPSClEnForCreator

class LAMMPSClEnForCalc(Calc):
    """
    Single-point energy calculation in LAMMPS, using a classical potential

    """

    def __init__(self):
        """
        Define calculation type and the corresponding method, engine and classes.

        """

        super().__init__(calc_type='energy_force',
                method='classical_potential', engine='LAMMPS',
                creator=LAMMPSClEnForCreator, runner=None, pp=None,
                description='Single-point energy calculation in LAMMPS, using a classical potential')
