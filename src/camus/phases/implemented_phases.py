import pkgutil
import inspect
import importlib
import click

import camus.phases

from collections import namedtuple

def get_implemented_phases():
    """
    Searches camus.phases to find classes which have
    ``phase_tag`` and ``phase_description`` class attributes.

    Returns
    -------
    phases : list
        List of named tuples of form namedtuple('phase', ('cls', 'tag', 'description'))

    """

    phases = []
    wf_tuple = namedtuple('phase', ('cls', 'tag', 'description'))

    for modules in pkgutil.walk_packages(path=camus.phases.__path__, prefix='camus.phases.'): 

        try:

            module = importlib.import_module(modules.name)

            for i, (cls_name, cls) in enumerate(inspect.getmembers(module, inspect.isclass)):
                if cls.__module__ == modules.name:
                    if (hasattr(cls, 'phase_tag') and hasattr(cls, 'phase_description')):

                        phases.append(wf_tuple(cls, cls.phase_tag, cls.phase_description))

        except:
            pass

    return phases

def get_wf_class_by_tag(tag, phases=None):
    """
    Find the phase with a given tag in ``phases`` and return the class.

    Parameters
    ----------
    tag : str
        Tag of the phase that is being searched for
    phases : None | list
        List of named tuples of form namedtuple('phase', ('class', 'tag', 'description')). If ``None``, get_implemented_phases is run.
    
    """

    if not phases:
        phases = get_implemented_phases()

    for phase in phases:

        if phase.tag == tag:

            wf_cls = phase.cls
            break

    else:
        click.echo('Invalid phase tag.')
        wf_cls = None

    return wf_cls

def print_implemented_phases(phases=None):
    """
    Prints "(n+1) phases[n].description" for n in range(0, len(phases)).

    Parameters
    ----------
    phases : None | list
        List of named tuples of form namedtuple('phase', ('class', 'tag', 'description')). If ``None``, get_implemented_phases is run.

    """

    if phases is None:
        phases = get_implemented_phases()

    for i, phase in enumerate(phases):
        click.echo(f'({i + 1}) {phase.description}')

if __name__  == '__main__':

    print_implemented_phases()
