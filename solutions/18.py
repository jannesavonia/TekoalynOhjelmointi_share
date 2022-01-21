import numpy as np

x=np.array([1, 2, 3])
y=np.array([3, 1, 0])

sum=0
for a, b in zip(x, y): 
    sum=sum+a*b

print(sum)
