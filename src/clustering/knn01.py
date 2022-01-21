import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm

filename='../orange/clustering01.csv'

#Load data
data=pandas.read_csv(filename, skiprows=[1, 2])
print("Read data shape = "+str(data.shape))
print()

classID=data["Class"]

#Remove index column and all text columns
for col in ["Class"]:
    print("Remove "+col)
    data.drop(col, axis=1, inplace=True)
print()

