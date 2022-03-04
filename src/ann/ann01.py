from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

n_train=10000
n_test=10000

def create_dataset(n):
    ret_x=2*np.random.rand(n, 2)-1
    r=np.sqrt(ret_x[:, 0]**2 + ret_x[:, 1]**2)
    ret_y=np.zeros((n, ), dtype=np.float64)
    print(np.shape(r))
    
    for i in range(n):
        if r[i]>0.2 and r[i]<0.4:
            ret_y[i]=-1
        elif r[i]>0.6 and r[i]<0.8:
            ret_y[i]=1
            
        if ret_x[i, 0]<0 and ret_x[i, 1]<0:
            ret_y[i]=-ret_y[i]
        
    return ret_x, ret_y

def plot(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, marker='.')
    plt.colorbar()

x_train, y_train = create_dataset(n_train)
plot(x_train, y_train)
plt.show()

x_test, y_test = create_dataset(n_test)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(2,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mse', optimizer='adam')

history = model.fit(
    x_train, y_train,
    epochs=300,
    verbose=1)

y_pred=model.predict(x_test)
plot(x_test, y_pred)
plt.show()

print("MSE =", mean_squared_error(y_test ,y_pred))

plt.plot(history.history['loss'])
plt.semilogy()
plt.show()