#Written by Bernard Hollands for F20DL CW 1

'''
9. Cluster  the  data  sets  train_smpl, train_smpl_<label>  (apply  required  filters and/or attribute selections if needed), using the k-means algorithm:
    •First  try  to  work  in  a  classical  clustering  scenario  and  assume  that  classes  are  not  given.  Research methods which allow you to visualise and analyse clusters (and the performance of the clustering algorithm on your data set).
    •Note the accuracy of k-means relative to the given clusters
'''
import sys
assert sys.version_info>=(3,5)

#Import sci-kit learn
import sklearn
assert sklearn.__version__>="0.20"
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
from sklearn.metrics import accuracy_score #to get an accuracy of the clustering methods
from sklearn.manifold import TSNE #to reduce the dimensionality of the data
from sklearn.cluster import KMeans #get accsess to kmeans clustering


# common imports
import numpy as np 
import os
import cv2
import pandas as pd
np.random.seed(50) #ensure the same result on every run

#to plot pretty sigures
#import matplotlib as mpl
import matplotlib.pyplot as plt

#Import all the data
def get_data_x(): #imports all the data from X_train_gr_random_reduced
    x_train_smpl = "x_train_gr_smpl_random_reduced.csv"#the csv file path
    x_data = pd.read_csv(x_train_smpl) #read the data from the file
    df1 = pd.DataFrame(x_data) #make it into a pandas dataframe
    x_data_array = df1.values#turn dataframe to numpy array
    x_train = x_data_array.astype('float') / 255 #normalise the data to be between 0 and 1
    x_train_sm, _ = train_test_split( #option of only using 20% of the dataset for faster testing
        x_train, test_size = 0.8
    )
    return x_train #return the data

def get_data_y():#get the y data aka all the the image classifications
    y_train_smpl = "y_train_smpl_random.csv"#the csv file path
    y_data = pd.read_csv(y_train_smpl)#read the data
    df2 = pd.DataFrame(y_data)#turn into dataframe
    y_data_array = df2.values
    y_train = y_data_array.astype('int') #make the data integers not floating points so can be used by accuracy_score
    y_train_sm, _ = train_test_split( #option of only using 20% of the dataset for faster testing
        y_train, test_size = 0.8
    )
    return y_train #return the data

def re_dimension(X, no_of_dim): #Method for reducing the dimensionaliy of the data
    x_train_low_dim = TSNE(n_components=no_of_dim, perplexity=35).fit_transform(X)#pass whatever data you want into the TSNE algorithum
    return x_train_low_dim #rreturn the data

def plot_data_2d(X): #Plots any 2 dimensional data that is passed ot it
    plt.scatter(X[:,0], X[:,1], alpha = 0.75, s = 4) #scatter each point according to what is in colmn 0 and 1
    plt.title("German Street Signs") #title the graph
    plt.show() #show the plot

#method was taken from tutorial 4 to plot the kmeans cluster centers/centriods
def plot_centroids(centroids, weights=None, circle_color='r', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1], #scatter the points that have been passed in from the centriods function of kmeans
                marker='x', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)


#performs the kmeans clustering
def K_Means(X, k):
    kmeans = KMeans(n_clusters = k, random_state=42) #define kmeans as a KMeans functions
    yhat = kmeans.fit_predict(X) #put the desired data X through the 
    clusters = np.unique(yhat) #creates clusters array of only the unique clusters 
    cluster_centers = kmeans.cluster_centers_ # get the center of the clusters
    plot_centroids(cluster_centers) #plot the centers
    for cluster in clusters: #look throught each cluster
        row_ix = np.where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1],s = 5) #plot the points
    print("KMeans Accuracy")
    y_train = get_data_y()#get the classification data
    print(accuracy_score(kmeans.predict(X), y_train)) # test the classification of KMeans vs the ones given
    plt.title("German Road Signs, KMeans Clustering") #title the plot
    plt.show() #the the kmeans clustering


def intertia_graph(X): #tests kmeans at different number of clusters
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X.astype('double'))
                    for k in range(1, 15)] #runs kmeans on the data 15 time 
    inertias = [model.inertia_ for model in kmeans_per_k] #gets the inertias for each run
    plt.plot(range(1, 15), inertias, "bo-") #for all 15 runs plot each intertia
    plt.xlabel("$k$", fontsize=14) #label x axis
    plt.ylabel("Inertia", fontsize=14)#label y axis
    plt.title("KMeans Interia for different k") #title the plot
    plt.axis([-1, 16, -1000, 800000]) #set size of x and y axis
    plt.show()#show the plot


def main(): #main method
    x_train = get_data_x() #gets the data
    x_train_2d = re_dimension(x_train ,2)#re-dimensions data to 2 dimensions
    plot_data_2d(x_train_2d)#plot the 2d data
    K_Means(x_train_2d, 10)#run kmeans clustering on the data with 10 clusters 
    intertia_graph(x_train_2d)#find get the inertia graph to find optimal number of clusters

if __name__ == "__main__":
    main()
