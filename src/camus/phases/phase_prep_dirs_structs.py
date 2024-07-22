from ase.io import read, write

from camus.phases.phase import Phase

class PhasePrepDirsStructs(Phase, phase_tag='prep_dirs_structs', phase_description='Create a directory and *.traj file for each given structure'):
    """
    Phase that takes an ASE-readable structure(s) file and creates a directory and .traj file for each structure.

    """

    _global_phase_config = [Phase._GlobalPhaseConfigOption(label='structures_file', description='Path to an ASE-readable structure(s) file.', allowed_values=None, default_value=None)]

    def __init__(self):
        """
        abcdef
        
        """
        pass
