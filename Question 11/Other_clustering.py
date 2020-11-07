#Written by Bernard Hollands for F20DL CW 1

#Written by Bernard Hollands for F20DL CW 1

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
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

from sklearn import mixture


# common imports
import numpy as np 
import os
import cv2
import pandas as pd
np.random.seed(42)

#to plot pretty sigures
import matplotlib as mpl
import matplotlib.pyplot as plt


#Import all the data

y_train_smpl = "x_train_gr_smpl_random_reduced.csv"#os.path.join(data_path, 'y_train_smpl_random.csv')#y data
x_train_smpl = "y_train_smpl_random.csv"#os.path.join(data_path, 'x_train_gr_smpl_random_reduced.csv') #x data

x_data = pd.read_csv(x_train_smpl) #read the data from the files
y_data = pd.read_csv(y_train_smpl)

df1 = pd.DataFrame(x_data)
df2 = pd.DataFrame(y_data)

x_data_array = df1.values#turn dataframe to numpy array
y_data_array = df2.values

x_train = x_data_array.astype('float') / 255
y_train = x_data_array.astype('float') /255

x_train_sm, _, y_train_sm, _  = train_test_split( #only using 5% of the dataset for speed
    x_train, y_train, test_size= .85
)


def re_dimension(X, no_of_dim):
    x_train_emb = TSNE(n_components=no_of_dim, perplexity=35).fit_transform(X)
    return x_train_emb

def plot_data_2d(X):
    plt.scatter(X[:,0], X[:,1], alpha = 0.75, s = 10)
    plt.title("German Street Signs")
    plt.legend()
    plt.show()


def Spectral_Clustering():
    x_train_2d = re_dimension(x_train,2)
    #transform data such that the distribution = 0 and std = 1
    model = SpectralClustering(n_clusters = 10)
    yhat = model.fit_predict(x_train_2d)
    clusters = np.unique(yhat)
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)
        plt.scatter(x_train_2d[row_ix, 0], x_train_2d[row_ix, 1])

    plt.title("Spectral Clustering")
    plt.show()




#scaled_X = scalar.transform(x_train_2d)

