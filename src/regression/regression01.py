import numpy as np
import matplotlib.pyplot as plt

N_meas=11

#real values
b_real=0.7
a_real=11.2

#generate measurement data
data_x=np.linspace(-5, 5, N_meas)
data_y=a_real+b_real*data_x+np.random.randn(N_meas)

#create matrices
X=np.ones((N_meas, 2))
X[:, 1]=data_x
Y=np.reshape(data_y, (N_meas, 1))

#compute a and b
ab=np.linalg.inv(X.T @ X) @ X.T @ Y
a_estimate=ab[0, 0]
b_estimate=ab[1, 0] 

#compute estimated values
y_estimate=a_estimate+b_estimate*data_x

#plot figure
fig=plt.figure(figsize=(16, 8))
plt.plot(data_x, data_y, '*')
plt.plot(data_x, y_estimate)
#plt.savefig("regression01.png")
plt.show()
