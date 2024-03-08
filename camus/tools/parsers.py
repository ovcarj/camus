""" 

Parsers of various output files.

"""

def parse_lammps_dump(specorder, log_lammps='log.lammps', dump_name='minimized.xyz'):
    """
    Reads a LAMMPS dump in `directory` and returns an ASE Atoms object with written forces and energies (which are read from `log_lammps`).
    """

    with open(dump_name) as f:
        structure = read_lammps_dump(f, specorder=specorder)

    # Get potential energy
    with open(log_lammps) as f:
        log_lines = f.readlines()

    for i, line in enumerate(log_lines):
        if 'Energy initial, next-to-last, final =' in line:
            energies_line = log_lines[i+1].strip()
            potential_energy = energies_line.split()[-1]
            structure.calc.results['energy'] = float(potential_energy)
            break

        elif 'Step PotEng' in line:
            energies_line = log_lines[i+1].strip()
            potential_energy = energies_line.split()[1]
            structure.calc.results['energy'] = float(potential_energy)
            break

    return structure


