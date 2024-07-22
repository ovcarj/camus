import abc

from camus.workflows.workflow import Workflow
from camus.phases.phase_prep_dirs_structs import PhasePrepDirsStructs

class WorkflowEnFor(Workflow,
        workflow_tag='energy_force', 
        workflow_description='Single point energy-force calculation(s)'
        ):
    """
    Single point energy-force calculation(s).

    """

    @classmethod
    def _define_required_wf_options(cls):
        """
        Defines ``cls._required_wf_options``.

        """

        cls._required_wf_options = None

    @classmethod
    def _get_phases(cls):
        """
        Returns a list of Phase subclasses used by WorkflowEnFor.

        Returns
        -------
        phases : list
            A list of Phase subclasses used by the workflow

        """

        phases = [PhasePrepDirsStructs]

        return phases

    def __init__(self, workflow_base_directory=None, workflow_config_path=None):
        """
        Sets the ``self._base_directory`` attribute and reads the workflow config file.

        Parameters
        ----------
        workflow_base_directory : None | str
            Path to the workflow base directory.
        workflow_config_path : None | str
            Path to the workflow config file.
        
        """

        super().__init__(workflow_base_directory=workflow_base_directory, 
                workflow_config_path=workflow_config_path)
