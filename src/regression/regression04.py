import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

N_meas=31

#real values
c_real=-0.3
b_real=0.7
a_real=11.2

#generate measurement data
data_x=np.linspace(-10, 10, N_meas)
data_y=a_real+b_real*data_x+c_real*(data_x**2)+3*np.random.randn(N_meas)

#convert to matrix
data_x=data_x.reshape((N_meas, 1))
data_y=data_y.reshape((N_meas, 1))

##############################
#create linear regression model
linregr = linear_model.LinearRegression()

linregr.fit(data_x, data_y)

y_estimate = linregr.predict(data_x)

print('linear b_estimate =', linregr.coef_[0,0])
print('linear a_estimate =', linregr.intercept_[0])
print()

##############################
#create polynomial regression model

#create X matrix (convert to linear form)
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(data_x)
print('X_poly =', X_poly[1:10, :])

#fit data to linear model
linreg2=linear_model.LinearRegression()
linreg2.fit(X_poly,data_y)

print('polyn c_estimate  =', linreg2.coef_[0,2])
print('polyn b_estimate  =', linreg2.coef_[0,1])
print('polyn a_estimate  =', linreg2.intercept_[0])
print()
print('c_real            =', c_real)
print('b_real            =', b_real)
print('a_real            =', a_real)

y_estimate2=linreg2.predict(poly_reg.fit_transform(data_x))

fig=plt.figure(figsize=(16, 8))
plt.plot(data_x, data_y, '*')
plt.plot(data_x, y_estimate, ':')
plt.plot(data_x, y_estimate2,color='blue')
plt.savefig("regression04.png")
plt.show()