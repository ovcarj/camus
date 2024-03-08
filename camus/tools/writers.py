""" 

Writers of various input files.

"""

from ase import Atoms
from ase.io import write

def write_lammps_data(input_structures=None, target_directory=None, prefixes='auto', specorder=None, write_masses=True,
        atom_style='atomic'):
    """ Creates LAMMPS data files from a list of ASE Atoms objects in a target directory.

    If `input_structures` is not given, self.structures are used.
    If `target_directory` is not given, CAMUS_LAMMPS_DATA_DIR environment variable is used.
    If `prefix='auto'`, a list of integers [0_, 1_, ...] will be used as prefixes of the file names.
    Otherwise, a list of prefixes should be given and the filenames will be `{prefix}lammps.data`.
    specorder: order of atom types in which to write the LAMMPS data file.
    If write_masses=True, Masses section will be added to the created file LAMMPS data file. In this case,
    """
    if input_structures is None:
        input_structures = self.structures
    if target_directory is None:
        target_directory = os.getenv('CAMUS_LAMMPS_DATA_DIR')

    # Create target directory if it does not exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Special case of single input structure:
    if isinstance(input_structures, Atoms): input_structures = [input_structures]

    # Write LAMMPS data files
    for i, structure in enumerate(input_structures):
        if prefixes == 'auto':
            prefix = str(i) + '_'
        else:
            if isinstance(prefixes, list):
                prefix = str(prefixes[i])
            elif isinstance(prefixes, str):
                prefix = prefixes

        file_name = os.path.join(target_directory, f'{prefix}lammps.data')
        write(file_name, structure, format='lammps-data', specorder=specorder, atom_style=atom_style)

        # Write masses if requested
        if write_masses:

            masses = []
            for spec in specorder:
                for atom in structure:
                    if atom.symbol == spec:
                        masses.append(atom.mass)
                        break
                    else: pass

            with open(file_name, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if line.strip() == 'Atoms':
                    lines.insert(i, 'Masses\n\n')
                    for j, spec in enumerate(specorder):
                        lines.insert(i+j+1, f'{j+1} {masses[j]} # {spec}\n')
                    lines.insert(i+j+2, '\n')
                    break

            with open(file_name, 'w') as f:
                f.writelines(lines)



