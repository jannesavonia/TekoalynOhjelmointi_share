import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

kmeans=KMeans(n_clusters=3).fit(data)

kmeans_classID=kmeans.labels_

plt.scatter(x=data['x'], y=data['y'], c=kmeans_classID)

#Optimimäärä ks. https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/


