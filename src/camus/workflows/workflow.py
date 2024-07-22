import abc
import click

from collections import namedtuple

from camus.cfg.config_workflow import ConfigWorkflow

class Workflow(abc.ABC):
    """
    Base Workflow class.

    """

    def __init_subclass__(cls, workflow_tag, workflow_description):
        """
        Ensure that every Workflow subclass defines required class attributes correctly.

        Parameters
        ----------
        workflow_tag : str
            Unique workflow identifier
        workflow_description : str
            A description of what the workflow does

        """
        
        cls.workflow_tag = workflow_tag
        cls.workflow_description = workflow_description

        cls._GlobalWfConfigOption = namedtuple('_GlobalWfConfigOption', ('label', 'description', 'allowed_values', 'default_value'))

        cls._define_general_wf_options()

        cls._define_required_wf_options()
        cls._check_required_wf_options()

    @classmethod
    def _define_general_wf_options(cls):
        """
        Define general configuration options common to all workflows as a list
        of _GlobalWfConfigOption instances.
        
        """

        cls._general_wf_options = [
                cls._GlobalWfConfigOption('workflow_run_mode', 'Should the workflow advance through the phases automatically or manually?', ['auto', 'manual'], 'manual')
                ]

    @classmethod
    @abc.abstractmethod
    def _define_required_wf_options(cls):
        """
        Each Workflow subclass must define ``cls._required_wf_options``,
        which must be None or a list of _GlobalWfConfigOption named tuples containing
        a label and a description. These options will be written into the
        main workflow options file by label.
        
        This has twofold possible usage:
        (1) Used by _get_phases(cls) to find the appropriate workflow phases
        (2) Can be used to define the global workflow logic, e.g., a parameter
        ``n_cycles``, defining how many times to loop through phases

        """
        pass

    @classmethod
    def _check_required_wf_options(cls):
        """
        Checks whether ``cls._required_wf_options is correctly implemented.

        """

        if not hasattr(cls, '_required_wf_options'):
            raise NotImplementedError('cls._required_wf_options not implemented in Workflow subclass.')

        else:

            assert (cls._required_wf_options is None) or (isinstance(cls._required_wf_options, list))

            if cls._required_wf_options:
                for cfg in cls._required_wf_options:
                    assert isinstance(cfg, cls._GlobalWfConfigOption)

    @classmethod
    @abc.abstractmethod
    def _get_phases(cls):
        """
        Returns a list of Phase subclasses used by the workflow.

        Returns
        -------
        phases : list
            A list of Phase subclasses used by the workflow

        """
        pass

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

        if workflow_base_directory:
            self._base_directory = base_directory

        if workflow_config_path:
            self._config = ConfigWorkflow(workflow_config_path=config_filepath)
            self.current_phase = self._config['WORKFLOW']['current_phase']
