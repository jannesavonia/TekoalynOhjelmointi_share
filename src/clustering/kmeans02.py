import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

n_clusters_array=[i for i in range(2, 10)]
silhouette_avg=[]

print('Compute silhouette scores')
for n_clusters in n_clusters_array: 
    kmeans=KMeans(n_clusters=n_clusters).fit(data)
    kmeans_classID=kmeans.labels_
 
    # silhouette score
    silhouette_avg.append(silhouette_score(data, kmeans_classID))
    
plt.plot(n_clusters_array,silhouette_avg)
#plt.scatter(x=data['x'], y=data['y'], c=kmeans_classID)
plt.show()

max_idx=silhouette_avg.index(max(silhouette_avg))
print("Recommended number of clusters is", n_clusters_array[max_idx])


