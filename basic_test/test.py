from ase.io import read
from camus import camus

structs = read('test.traj', index='0:100')

camus_object = camus.Camus(structs)

print(f'Read {len(camus_object.Cstructures.structures)} structures into a camus.Cstructures object.')

camus_object.Cstructures.create_datasets()

print(f'Created ML datasets. {len(camus_object.Cstructures.test_set)} structures are in the test set.')

empty = camus.Camus(structs)

print(f'Created camus object without any structures.')

empty.Cstructures.structures = structs

print(f'Added {len(empty.Cstructures.structures)} structures to the empty camus object.')

print('Test OK')
