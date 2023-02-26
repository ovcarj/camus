""" Definition of the ARTn class.

This module defines everything related to handling ARTn inputs and outputs. 

"""

class ARTn:

    def __init__(self, artn_outputs=[]):
        """
        Initializes a new ARTn object. 

        Parameters:
            artn_outputs (list of ARTn output filenames): TODO: user should be able to parse these outputs.  
                Defaults to an empty list.
        """

        self._artn_outputs = artn_outputs

    @property
    def artn_outputs(self):
        return self._artn_outputs

    @artn_outputs.setter
    def artn_outputs(self, new_artn_outputs):
        self._artn_outputs = new_artn_outputs

    @artn_outputs.deleter
    def artn_outputs(self):
        del self._artn_outputs

    def write_artn_in(self):
        """ Method that writes a standard artn.in file to a `target directory`. 
        If a `target directory` is not specified, CAMUS_ARTN_DATA_DIR environment variable is used.

        Parameters:
            parameter: parameter description 
 
        """

        # Set the default target directory to CAMUS_ARTN_DATA_DIR environment variable
        if target_dir is None:
            target_dir = os.environ.get('CAMUS_ARTN_DATA_DIR')
            if target_dir is None:
                raise ValueError("Target directory not specified and CAMUS_ARTN_DATA_DIR environment variable is not set.")

