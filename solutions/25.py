import numpy as np

A=np.array([[1, 2, 3],
            [2, 1, 5],
            [4, 2, 3]])

shape=np.shape(A)

I=np.eye(shape[0], shape[1])

AI=A@I

if (AI==A).all():
    print('On sama')
else:
    print('Ei ole sama')
