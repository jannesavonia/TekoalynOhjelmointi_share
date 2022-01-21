import numpy as np
from sklearn import linear_model
import pickle

N_meas=200
#Input R^3, output R^2

#real values
b_real=np.array([[1.8, -0.5, -1], [-0.4, 8.0, 2]])
a_real=np.array([11.2, 0.2])

n_dim_in=np.shape(b_real)[1]
n_dim_out=np.shape(b_real)[0]

#generate measurement data
data_x=10*(np.random.rand(N_meas, n_dim_in)-0.5)
data_y=a_real+data_x@b_real.T+np.random.randn(N_meas, n_dim_out)

#create linear regression model
regr = linear_model.LinearRegression()

#train the model using the training set
regr.fit(data_x, data_y)

print("y=a+x@b.T")
print("input shape  =", np.shape(data_x))
print("output shape =", np.shape(data_y))
print("b shape      =", np.shape(regr.coef_))
print("a shape      =", np.shape(regr.intercept_))
print()

print('b_estimate =\n', regr.coef_, '\nb_real =\n', b_real)
print()
print('a_estimate =\n', regr.intercept_, '\na_real =\n', a_real)
print()

# save the model to disk with pickle
filename = 'regression03_model.sav'
pickle.dump(regr, open(filename, 'wb'))

del regr, data_x
 
# load the model from disk
regr_loaded = pickle.load(open(filename, 'rb'))

#predict values
data_x=10*(np.random.rand(N_meas, n_dim_in)-0.5)
y_estimate = regr_loaded.predict(data_x)
print('b_loaded =\n', regr_loaded.coef_, '\nb_real =\n', b_real)
print()
print('a_loaded =\n', regr_loaded.intercept_, '\na_real =\n', a_real)
print()

