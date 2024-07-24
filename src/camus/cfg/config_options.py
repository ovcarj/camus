from collections import namedtuple

class ConfigOptions:
    """
    Define the scope and sanitize configuration options.

    """

    def __init__(self):
        """
        Initialize ``self.options`` list and self._option named tuple.

        """

        self.options = []

        self._CfgOption = namedtuple('_CfgOption', 
                ('section', 'key', 'cfg_type', 'description', 'message', 
                'scope', 'value', 'allowed_values', 'default_value'),
                defaults=(None, None, None, None))

    def add_option(self, 
            section, key, cfg_type, description, message,
            scope=None, value=None, allowed_values=None, default_value=None):
        """
        Append a ``_CfgOption`` named tuple to ``self.options``.

        Parameters
        ----------

        section : str
            Section in the configuration file
        key : str
            Key in the configuration file
        cfg_type : str
            One of the following: meta, env, sched, logic
        description : str
            Description of the configuration option
        message : None | str
            Message to be printed to the user during the configuration process
        scope : None | str
            One of the following: global, project, workflow, phase, calc
        value : None | str
            Value of the configuration option
        allowed_values : None | list
            List of strings of allowed configuration options. If None, any value is allowed
        default_value : None | str
            Default value of the configuration option.
            
        """

        pass
