import os

from ase.io import read
from camus import camus

structs = read('test.traj', index='0:20')
candidate_structs = read('test.traj', index='15:35')
candidate_structs2 = read('test.traj', index='515:525')

camus_object = camus.Camus(structures=structs)

print(f'Read {len(camus_object.Cstructures.structures)} structures into a camus.Cstructures object.')

camus_object.Cstructures.create_datasets()

print(f'Created ML datasets. {len(camus_object.Cstructures.test_set)} structures are in the test set.')

camus_object.Cstructures.create_sisyphus_set()

print(f'Created Sisyphus set randomly. {len(camus_object.Cstructures.sisyphus_set)} structures are in the Sisyphus set.')

#camus_object.Cstructures.calculate_descriptors()

#print(f'Created {len(camus_object.Cstructures.descriptors)} default descriptors.')

candidate_structures = camus.Structures(structures=candidate_structs)
candidate_structures2 = camus.Structures(structures=candidate_structs2)

unique_structs = camus_object.Cstructures.find_unique_structures(reference_set_structures=camus_object.Cstructures, candidate_set_structures=candidate_structures)
unique_structs2 = camus_object.Cstructures.find_unique_structures(reference_set_structures=camus_object.Cstructures, candidate_set_structures=candidate_structures2)

print(f'Found {len(unique_structs)} unique structures.')
print(f'Found {len(unique_structs2)} unique structures.')

empty = camus.Camus(structs)

print(f'Created camus object without any structures.')

empty.Cstructures.structures = structs


print(f'Added {len(empty.Cstructures.structures)} structures to the empty camus object.')

# Test non-standard LAMMPS input writing
# Coeffs from https://doi.org/10.1039/D0TA03200J

lmp_input_dict = {
        'units': 'real',
        'atom_style': 'charge',
        'boundary': 'p p p',
        'read_data': 'lammps.data',
        'labelmap': 'atom 1 Br 2 I 3 Cs 4 Pb',
        'pair_style': 'buck/coul/long 9.0',
        'kspace_style': 'pppm 1.0e-7',
        'pair_coeff': ['Pb Pb 0.0 0.25 1024.0',
         'Pb Cs 8753.0 0.4187 1.108',
         'Pb I 8955.0 0.3040 267.5',
         'Pb Br 6447.0 0.3010 330.9',
         'Cs Cs 7512.0 0.04378 227.8',
         'Cs I 40322.0 0.3310 0.0',
         'Cs Br 9345.0 0.3331 117.5',
         'I I 62090.0 0.2752 1022.0',
         'I Br 151700.0 0.2663 918.4',
         'Br Br 370700.0 0.2420 277.9'],
        'compute': 'cpe all pe',
        }

camus_object.Csisyphus.set_lammps_parameters(input_parameters=lmp_input_dict)
camus_object.Csisyphus.write_lammps_in(filename='test_nonstd_lmp.in')

print('Wrote non-standard LAMMPS input to $CAMUS_LAMMPS_DATA_DIR.')

camus_object.Csisyphus.lammps_parameters = {}

sisyphus_test_directory = os.path.join(os.environ.get('CAMUS_SISYPHUS_DATA_DIR'), 'sisyphus_test_dir')
camus_object.create_sisyphus_calculation(target_directory=sisyphus_test_directory, specorder=['Br', 'I', 'Cs', 'Pb'])

print(f'Created files for a Sisyphus calculation in {sisyphus_test_directory}')

camus_object.Csisyphus.lammps_parameters = {}

minimization_test_directory = os.path.join(os.environ.get('CAMUS_LAMMPS_MINIMIZATION_DIR'), 'minimization_test_dir')
camus_object.create_lammps_minimization(target_directory=minimization_test_directory, specorder=['Br', 'I', 'Cs', 'Pb'])

print(f'Created files for a LAMMPS minimization in {minimization_test_directory}')

camus_object.Csisyphus.lammps_parameters = {}
batch_minimization_test_directory = os.path.join(os.environ.get('CAMUS_LAMMPS_MINIMIZATION_DIR'), 'batch_minimization_test_dir')
camus_object.create_batch_minimization(base_directory=batch_minimization_test_directory, specorder=['Br', 'I', 'Cs', 'Pb'])

print(f'Created files for a batch LAMMPS minimization in {batch_minimization_test_directory}')

#camus_object.Cdft.dft_parameters = {}

#batch_dft_directory = os.path.join(os.environ.get('CAMUS_DFT_DIR'), 'dft_test_dir')
#camus_object.create_batch_dft(base_directory=batch_dft_directory, input_structures=candidate_structs)

#print(f'Created files for batch DFT calculation in {batch_dft_directory}')

print('Test OK')
