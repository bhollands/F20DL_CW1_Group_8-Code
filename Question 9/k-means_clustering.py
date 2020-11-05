#Written by Bernard Hollands
#https://blog.galvanize.com/introduction-k-means-cluster-analysis/
'''
9. Cluster  the  data  sets  train_smpl, train_smpl_<label>  (apply  required  filters and/or attribute selections if needed), using the k-means algorithm:
    •First  try  to  work  in  a  classical  clustering  scenario  and  assume  that  classes  are  not  given.  Research methods which allow you to visualise and analyse clusters (and the performance of the clustering algorithm on your data set).
    •Note the accuracy of k-means relative to the given clusters
'''
import sys
assert sys.version_info>=(3,5)
import sklearn
assert sklearn.__version__>="0.20"
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# common imports
import numpy as np 
import os
import cv2
import pandas as pd
np.random.seed(42)
#to plot pretty sigures
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Import all the data

y_train_smpl = "x_train_gr_smpl_random_reduced.csv"#os.path.join(data_path, 'y_train_smpl_random.csv')#y data
x_train_smpl = "y_train_smpl_random.csv"#os.path.join(data_path, 'x_train_gr_smpl_random_reduced.csv') #x data

x_data = pd.read_csv(x_train_smpl)
y_data = pd.read_csv(y_train_smpl)

df1 = pd.DataFrame(x_data)
df2 = pd.DataFrame(y_data)

x_data_array = df1.values
y_data_array = df2.values

x_train = x_data_array.astype('float') / 255
y_train = x_data_array.astype('float') /255

x_train_sm, _, y_train_sm, _  = train_test_split(
    x_train, y_train, test_size= .95
)


def re_dimension(X, no_of_dim):
    x_train_emb3 = TSNE(n_components=no_of_dim, perplexity=35).fit_transform(X)
    return x_train_emb3

def plot_clusters_3d(X):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0], X[:,1], X[:,2], alpha = 0.75)
    ax.legend
    #plt.show()

def plot_clusters_2d(X):
    plt.scatter(X[:,0], X[:1], alpha = 0.75)
'''
for label in range(10):
    print(y_train_sm == label)
    x_train_tmp = x_train_emb3#[y_train_sm == label]
   ''' 

x_train_emb3 = re_dimension(x_train_sm,3)
plot_clusters_3d(x_train_emb3)

#x_train_emb2 = re_dimension(x_train_sm,2)
#plot_clusters_2d(x_train_emb2)


k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(x_train_emb3)
y_pred
y_pred is kmeans.labels_

print(kmeans.cluster_centers_)

