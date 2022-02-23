import pandas 
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

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


print("Create a dendrogram")
linked = linkage(data, 'single')
labelList = range(len(data))

dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

print("Select n_clusters=3")
n_clusters=3
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.scatter(x=data['x'], y=data['y'], c=classID)
plt.show()



