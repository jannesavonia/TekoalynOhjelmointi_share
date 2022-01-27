# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:46:27 2022

@author: kojape
"""

import numpy as np
import matplotlib.pyplot as plt

def plotVector(O, x, c):
    plt.arrow(O[0], O[1], x[0], x[1], head_width=0.1, head_starts_at_zero=False, head_length=0.1, fc=c, ec=c)

def plotLine(O, x, c):
    plt.plot([O[0], x[0]], [O[1], x[1]], c)

O=np.array([0,0])
x=np.array([-1,4])
y=np.array([-4, 1])

A=np.array([[0.5, 1],
            [-1, 3]])

fig=plt.figure(figsize=(8,8))

"""
#matrix multiplication
plotVector(O, x, 'r')
Ax=A@x
plotVector(O, Ax, 'm')

plotVector(O, y, 'b')
Ay=A@y
plotVector(O, Ay, 'c')
"""

#unitsphere transform
phi_array=np.arange(0, 2*np.pi, 0.4)
for phi in phi_array:
    z=np.array([np.cos(phi),np.sin(phi)])
    Az=A@z
    plotVector(z, Az, 'k')


"""
#Create unit vector
plotVector(O, x, 'r')
nx=np.linalg.norm(x)
hat_x=(1/nx)*x
plotVector(O, hat_x, 'k')
"""

"""
#Add two vectors
plotVector(O, x, 'r')
plotVector(O, y, 'b')
plotVector(x, y, 'b')
plotVector(O, x+y, 'k')
"""


"""
#Subtract vetors
plotVector(O, x, 'r')
plotVector(O, y, 'b')
plotVector(x, -y, 'b')
plotVector(O, x-y, 'k')
"""

"""
#dot product and perpendicular vectors
x_=np.array([x[1], -x[0]])
plotVector(O, x, 'r')
plotVector(O, x_, 'b')
print('x . x_ =', x.dot(x_))
"""


"""
#dot product and not perpendicular vectors
x_=np.array([x[1], -x[0]])
plotVector(O, x, 'r')
plotVector(O, y, 'b')
print('x . y =', x.dot(y))
"""


"""
#projection
plotVector(O, x, 'r')
plotVector(O, y, 'b')

# Lasketaan x:n projektio y:lle
Py=((x.dot(y))/(y.dot(y)))*y
plotVector(O, Py, 'c')
plotLine(Py, x, 'k--')
"""

plt.xlim([-6, 6])
plt.ylim([-6, 6])

plt.show()

