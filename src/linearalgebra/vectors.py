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
y=np.array([-3, 1])

fig=plt.figure(figsize=(8,8))

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
Py=((y.dot(x))/(y.dot(y)))*y
plotVector(O, Py, 'k')
plotLine(Py, x, 'k--')
"""

plt.xlim([-6, 6])
plt.ylim([-6, 6])

plt.show()

