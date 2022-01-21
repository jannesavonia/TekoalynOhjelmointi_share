import pandas
from scipy import stats
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

#Original data: https://www.kaggle.com/quantbruce/real-estate-price-prediction/version/1
#Data modified by Janne Koponen

filename='RealEstate.csv'

#Load data
data=pandas.read_csv(filename)
print("Read data shape = "+str(data.shape))
print()

#Remove index column
print("Remove index column")
data.drop(data.columns[0], axis=1, inplace=True)
print("data shape = "+str(data.shape))
print()

#Show lines with missing values
print("Lines with missing valaues:")
print(data[data.isnull().any(axis=1)])
print()

#Remove missing values
print("Remove missing values")
data.dropna(inplace=True)
print("data shape = "+str(data.shape))
print()

#Remove duplicates
print("Remove duplicates")
data.drop_duplicates(inplace=True)
print("data shape = "+str(data.shape))
print()

#Find outliers by z-test
print("Find outliers by z-test")
z_threshold=6.0 #3 is recommended for most cases

z = np.abs(stats.zscore(data))
outlier=(z>=z_threshold)
print(data[outlier])

filtered_entries = (z < z_threshold).all(axis=1)

#Remove outliers
print("Remove outliers")
data = data[filtered_entries]
print(data)
print()

#Split training set and test set
train_fraction=0.8

print("Create training data set")
traindata=data.sample(frac=train_fraction,random_state=200) #random state is a seed value
print("data shape = "+str(traindata.shape))
print()

print("Create test data set")
testdata=data.drop(traindata.index)
print("data shape = "+str(testdata.shape))
print()

#Convert dataframe to numpy array
print("Convert df to array")
traindata=traindata.to_numpy()
print(traindata.shape)
print()

#Standardize data, see https://builtin.com/data-science/step-step-explanation-principal-component-analysis
#print("Standardize data")
#stdev=traindata.std(axis=0)
#mean=traindata.mean(axis=0)

#traindata=(traindata-mean)/stdev
#print(traindata.shape)
#print(traindata)
#print()

#Linear regression
print("Create training X and Y")

def splitXY(data):
    return data[:, :-1], data[:, -1:]

trainX, trainY=splitXY(traindata)
print(trainX.shape, trainY.shape)

# Create linear regression object
print("Create linear regression model")
regr = linear_model.LinearRegression()

# Train the model using the training sets
print("Train the model")
regr.fit(trainX, trainY)

#Create test x and test y
print("Create test X and Y")
testdata=testdata.to_numpy()
print(testdata.shape)
testX, testY=splitXY(testdata)
print(testX.shape, testY.shape)

#Predict values
predY = regr.predict(testX)

#Plot prediction
plt.plot(testY, predY, '.')
plt.plot([10, 70], [10, 70])
plt.xlabel("real value")
plt.ylabel("predicted value")
plt.show()