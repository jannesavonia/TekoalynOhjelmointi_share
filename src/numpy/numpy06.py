import numpy as np

a1=np.array([[0.1,  0.4,  0.4],
             [0.0,  1.0, -1.0]], dtype=np.float64)

print('Taulukko on', len(a1.shape), 'ulotteinen')
print('Taulukon koko on', a1.shape)
print('Taulukossa on', a1.size, 'alkiota')
print('Taulukon alkio on tyyppia', a1.dtype)
