import abc

from collections import namedtuple

class Phase(abc.ABC):
    """
    Base Phase class.

    """

    def __init_subclass__(cls, phase_tag, phase_description):
        """
        Ensure that every Phase subclass defines required class attributes correctly.

        Parameters
        ----------
        phase_tag : str
            Unique phase identifier
        phase_description : str
            A description of what the phase does

        """

        cls.phase_tag = phase_tag
        cls.phase_description = phase_description

        cls._GlobalPhaseConfigOption = namedtuple('_GlobalPhaseConfigOption', ('label', 'description', 'allowed_values', 'default_value'))

        cls._define_general_phase_options()

        cls._define_required_phase_options()
        cls._check_required_phase_options()

    def __init__(self):
        """
        Phase initialization...
        
        """
        pass
