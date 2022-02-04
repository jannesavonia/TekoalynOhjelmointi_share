import matplotlib.pyplot as plt
import numpy as np

myynti1=[26, 43, 22, 33, 9, 1, 9, 8, 33, 11, 32, 7]
myynti2=[16, 23, 21, 13, 19, 11, 39, 28, 23, 21, 23, 17]

kk=['tammi', 'helmi', 'maalis', 'huhti', 'touko', 'kesä', 'heinä', 'elo', 'syys', 'loka', 'marras', 'joulu']


width=0.35
x=np.arange(len(myynti1))

fig, ax=plt.subplots()
plt.bar(x-width/2, myynti1, width=width, label='myynti1')
plt.bar(x+width/2, myynti2, width=width, label='myynti2')
ax.set_xticks(x)
ax.set_xticklabels(kk, rotation=45)
ax.legend()
plt.show()

