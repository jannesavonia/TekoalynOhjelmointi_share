import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

N_meas=11

#real values
b_real=0.7
a_real=11.2

#generate measurement data
data_x=np.linspace(-5, 5, N_meas)
data_y=a_real+b_real*data_x+np.random.randn(N_meas)

#convert to matrix
data_x=data_x.reshape((N_meas, 1))
data_y=data_y.reshape((N_meas, 1))

#create linear regression model
regr = linear_model.LinearRegression()

#train the model using the training set
regr.fit(data_x, data_y)

#predict values
y_estimate = regr.predict(data_x)

print('b_estimate =', regr.coef_[0,0])
print('a_estimate =', regr.intercept_[0])

#plot figure
fig=plt.figure(figsize=(16, 8))
plt.plot(data_x, data_y, '*')
plt.plot(data_x, y_estimate)
#plt.savefig("regression02.png")
plt.show()
