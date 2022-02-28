from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

n_train=10000
n_test=1000

def create_dataset(n):
    ret_x=2*np.random.rand(n, 2)-1
    r=np.sqrt(ret_x[:, 0]**2 + ret_x[:, 1]**2)
    ret_y=np.zeros((n, ), dtype=np.float64)
    
    print(np.shape(r))
    
    for i in range(n):
        if (r[i]>0.2 and r[i]<0.4) or (r[i]>0.6 and r[i]<0.8):
            ret_y[i]=1  
    
    return ret_x, ret_y

def plot(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, marker='.')

x_train, y_train = create_dataset(n_train)
plot(x_train, y_train)
plt.show()

x_test, y_test = create_dataset(n_test)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = np_utils.to_categorical(y_train, num_classes)
#y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(2,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mse', optimizer='adam')

history = model.fit(
    x_train, y_train,
    epochs=30,
    verbose=1)

y_pred=model.predict(x_test)
y_pred_int=np.round(y_pred).reshape(y_test.shape) #convert to 0/1 value
plot(x_test, y_pred_int)
plt.show()

print("Faulty predictions:", np.abs(np.abs(y_test)-np.abs(y_pred_int)).sum())

