import pkgutil
import inspect
import importlib
import click

import camus.workflows

from collections import namedtuple

def get_implemented_workflows():
    """
    Searches camus.workflows to find classes which have
    ``workflow_tag`` and ``workflow_description`` class attributes.

    Returns
    -------
    workflows : list
        List of named tuples of form namedtuple('workflow', ('cls', 'tag', 'description'))

    """

    workflows = []
    wf_tuple = namedtuple('workflow', ('cls', 'tag', 'description'))

    for modules in pkgutil.walk_packages(path=camus.workflows.__path__, prefix='camus.workflows.'): 

        try:

            module = importlib.import_module(modules.name)

            for i, (cls_name, cls) in enumerate(inspect.getmembers(module, inspect.isclass)):
                if cls.__module__ == modules.name:
                    if (hasattr(cls, 'workflow_tag') and hasattr(cls, 'workflow_description')):

                        workflows.append(wf_tuple(cls, cls.workflow_tag, cls.workflow_description))

        except:
            pass

    return workflows

def get_wf_class_by_tag(tag, workflows=None):
    """
    Find the workflow with a given tag in ``workflows`` and return the class.

    Parameters
    ----------
    tag : str
        Tag of the workflow that is being searched for
    workflows : None | list
        List of named tuples of form namedtuple('workflow', ('class', 'tag', 'description')). If ``None``, get_implemented_workflows is run.
    
    """

    if not workflows:
        workflows = get_implemented_workflows()

    for workflow in workflows:

        if workflow.tag == tag:

            wf_cls = workflow.cls
            break

    else:
        click.echo('Invalid workflow tag.')
        wf_cls = None

    return wf_cls

def print_implemented_workflows(workflows=None):
    """
    Prints "(n+1) workflows[n].description" for n in range(0, len(workflows)).

    Parameters
    ----------
    workflows : None | list
        List of named tuples of form namedtuple('workflow', ('class', 'tag', 'description')). If ``None``, get_implemented_workflows is run.

    """

    if workflows is None:
        workflows = get_implemented_workflows()

    for i, workflow in enumerate(workflows):
        click.echo(f'({i + 1}) {workflow.description}')

if __name__  == '__main__':

    print_implemented_workflows()
