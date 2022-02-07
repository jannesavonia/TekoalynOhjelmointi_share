import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import linear_model


#http://mreed.umtri.umich.edu/mreed/downloads.html
#Male dataset!
#We have preprocessed data set which does not contain missing values nor duplicates
filename="../../../share/data/ANSUR2_2012/ANSUR II MALE Public.csv"
train_fraction = 0.8
Y_column='Weightlbs'

#Load data
data=pandas.read_csv(filename, encoding = "ISO-8859-1")
print("Read data shape = "+str(data.shape))
print()

#Remove index column, all text columns, and weightkg
for col in ["weightkg", "Gender","Date","Installation","Component","Branch","PrimaryMOS","SubjectsBirthLocation","SubjectNumericRace","Ethnicity","DODRace","WritingPreference","subjectid"]:
    print("Remove "+col)
    data.drop(col, axis=1, inplace=True)

print("data shape = "+str(data.shape))
print()

def splitXY(d):
    return d.drop(Y_column, axis=1), d[Y_column] 

print("Create training data set")
traindata=data.sample(frac=train_fraction,random_state=200)
trainX, trainY=splitXY(traindata)

print("Create test data set")
testdata=data.drop(traindata.index)
testX, testY=splitXY(testdata)

#Remove unused datasets
del data, testdata, traindata

print("trainX shape = "+str(trainX.shape))
print("trainY shape = "+str(trainY.shape))
print("testX shape = "+str(testX.shape))
print("testY shape = "+str(testY.shape))
print()


#Compute full PCA
print("Compute full PCA")
pca = PCA()
pca.fit(trainX)

plt.plot(pca.explained_variance_)
plt.show()

plt.plot(pca.explained_variance_)
plt.semilogy()
plt.show()

#PCA with reduced dimension, try values 1, 2, 5, 10
for packed_dimension in [1, 2, 5, 50]:
    print(30*"-")
    print("Compute reduced dimension PCA, n =", packed_dimension)
    
    del pca
    pca = PCA(packed_dimension)
    pca.fit(trainX)
    
    pca_trainX=pca.transform(trainX)
    pca_testX=pca.transform(testX)
    print("pca_trainX shape = "+str(pca_trainX.shape))
    print("pca_testX shape = "+str(pca_testX.shape))
    print()
    
    
    print("Create linear regression model")
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    print("Train the model")
    regr.fit(pca_trainX, trainY)
    
    #Predict values
    print("Predict values")
    predY = regr.predict(pca_testX)
    
    plt.plot(testY, predY, '.', label=str(packed_dimension))

plt.plot([90, 340],[90, 340], 'k:')
plt.xlabel("Weight (lbs)")
plt.ylabel("Predicted weight (lbs)")
plt.legend()
plt.show()
