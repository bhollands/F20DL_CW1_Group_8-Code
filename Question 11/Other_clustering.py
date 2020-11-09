#Written by Bernard Hollands for F20DL CW 1

import sys
assert sys.version_info>=(3,5)
import sklearn
assert sklearn.__version__>="0.20"
from sklearn.model_selection import train_test_split

#clustering imports
from sklearn.cluster import SpectralClustering #import spectral clustering
from sklearn.mixture import GaussianMixture #import the Gaussian Mixture Model
from sklearn.metrics import accuracy_score #import the accuracy score
from sklearn.manifold import TSNE #to reduce the dimensionality of the data

# common imports
import numpy as np 
import os
import cv2
import pandas as pd
np.random.seed(42)

#to plot pretty sigures
import matplotlib.pyplot as plt


#Import all the data

x_train_smpl = "x_train_gr_smpl_random_reduced.csv"# file path of the data
y_train_smpl = "y_train_smpl_random.csv" #file path for the classification data

x_data = pd.read_csv(x_train_smpl) #read the data from the files
y_data = pd.read_csv(y_train_smpl) #read the classification file

df1 = pd.DataFrame(x_data) # turn into a pandas dataframe
df2 = pd.DataFrame(y_data) # turn into a pandas dataframe

#print(df2)
x_data_array = df1.values #turn dataframe to numpy array
y_data_array = df2.values 


x_train = x_data_array.astype('float') / 255 #normalise image data to be between 0 and 1
y_train = y_data_array.astype('int') #turn classification data into an interger for accuracy_score

x_train_sm, _, y_train_sm, _  = train_test_split( #option to only use any% of the data for quicker testing
    x_train, y_train, test_size= .95
)

#modified from tutorial 4
def plot_centroids(centroids, weights=None, circle_color='r', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
        

def re_dimension(X, no_of_dim): #method for reducing the dimensioanlty of data
    x_train_low_dim = TSNE(n_components=no_of_dim, perplexity=35).fit_transform(X) #pass teh data through TSNE algorithum
    return x_train_low_dim

def plot_data_2d(X): #plot any 2d data on a scatter plot
    plt.scatter(X[:,0], X[:,1], alpha = 0.75, s = 10)
    plt.title("German Street Signs")
    plt.legend()
    plt.show()


def Spectral_Clustering(X, k):
    model = SpectralClustering(n_clusters = k) #make the model precral clustering with k clusters
    yhat = model.fit_predict(X) #run the data on the spectral cluster
    clusters = np.unique(yhat) #make an array of only unique clusters
    for cluster in clusters:# loop throgh all clusters and plot them
        row_ix = np.where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s = 10)
    plt.title("Spectral Clustering") #title the plot
    plt.show() #show the plot

def Gaussian_Mixture_model(X, k):
    model = GaussianMixture(n_components=k) #assign model to the gaussian mixture
    model.fit_predict(X) #run the data on the GMM
    yhat = model.predict(X) #assign yhat the labeled points
    clusters = np.unique(yhat) #make array of only unique clusters
    for cluster in clusters: #loop throught eclusters and plot them
        row_ix = np.where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s= 5)
    
    print("GMM Accuracy")
    print(accuracy_score(yhat, y_train)) # calcuate the accuracy of the clustering
    plt.title("Gaussians Mixture Model")
    plt.show()



def main(): #main method to run each cluster
    x_train_2d = re_dimension(x_train_sm,2)
    Gaussian_Mixture_model(x_train_2d, 10)
    Spectral_Clustering(x_train_2d, 10)

if __name__ == "__main__":
    main()





