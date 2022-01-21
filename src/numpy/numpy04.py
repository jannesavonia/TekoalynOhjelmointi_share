import numpy as np

filename='np_example_04.npz'

a1=np.array([0.1, 0.4, 0.4], dtype=np.float64)
a2=np.array(['a', 'b', 'c', 'd'])
z=np.array([0, 3, 8])
np.savez(filename, my_named_parameter=a1, a2=a2, z=z)

del a1, a2, z

npz_file=np.load(filename)
a1=npz_file['my_named_parameter']
a2=npz_file['a2']
z=npz_file['z']

