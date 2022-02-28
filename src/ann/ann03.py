import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
#https://archive.ics.uci.edu/ml/machine-learning-databases/wine/


data=pd.read_csv('wine.csv', header=None)
data=shuffle(data)

train_fraction=0.8

Y_column=0

def splitXY(d):
    return d.drop(Y_column, axis=1), d[Y_column] 

print("Split training/test data set")
traindata, testdata = train_test_split(data, test_size=1-train_fraction)
trainX, trainY=splitXY(traindata)
testX, testY=splitXY(testdata)

del data, testdata, traindata
print()

print("Standardize data")
stdev=trainX.std(axis=0)
mean=trainX.mean(axis=0)
trainX=(trainX-mean)/stdev
testX=(testX-mean)/stdev
print()

trainX=trainX.to_numpy()
testX=testX.to_numpy()
trainY=trainY.to_numpy()
testY=testY.to_numpy()
minY=np.min(trainY)
trainY=trainY-minY
testY=testY-minY

num_classes=np.max(trainY)+1
print("num_classes =", num_classes)


n_train, width_train=np.shape(trainX)
n_test, width_test=np.shape(testX)

# convert class vectors to binary class matrices
trainY = np_utils.to_categorical(trainY, num_classes)

assert(width_train==width_test)

trainY=trainY.reshape((n_train, num_classes))

print("trainX shape = "+str(trainX.shape))
print("trainY shape = "+str(trainY.shape))
print("testX shape = "+str(testX.shape))
print("testY shape = "+str(testY.shape))
print()


model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(width_train,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(trainX, trainY, epochs=30)

predY=model.predict(testX)
predY_int=np.round(predY) #convert to 0/1 value

#convert to values
predY_int=predY_int.argmax(1)

print("Faulty predictions:", np.abs(testY-predY_int).sum())
