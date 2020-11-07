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
    #plt.show()



def plot_centroids(centriods, weights=None, circle_color='r', cross_color='k'):
    if weights is not None:
        centriods = centriods[weights > weights.max() / 10]
    plt.scatter(centriods[:, 0], centriods[:, 1],
    marker='x', s=60, linewidths=2, color=circle_color, zorder=5, alpha=0.9)
    #plt.scatter(centriods[:, 0], centriods[:, 1],
    #marker='x', s=10, linewidths=2, color=circle_color, zorder=11, alpha=1)

#Function taken from tutorial
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centriods=True, 
                                show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    '''
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))

    z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(z, extent=(mins[0], maxs[0], mins[1],maxs[1]), cmap='Pastel2')
    plt.contour(z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')
    '''
    plot_data_2d(X)
    if show_centriods:
        plot_centroids(clusterer.cluster_centers_)
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def plot_boundaries_graph(X, clusterer):
    plt.figure(figsize=(8,4))
    plot_decision_boundaries(clusterer, X)  
    plt.show()

x_train_emb2 = re_dimension(x_train,2)


k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(x_train_emb2.astype('double'))
y_pred
y_pred is kmeans.labels_


plot_data_2d(x_train_emb2)
plot_boundaries_graph(x_train_emb2, kmeans)

print(kmeans.inertia_)
X_dist = kmeans.transform(x_train_emb2.astype('double'))
np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_]**2)
print(kmeans.score(x_train_emb2.astype('double')))

##Finding optimal clusters

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(x_train_emb2.astype('double'))
                for k in range(1, 15)]
inertias = [model.inertia_ for model in kmeans_per_k]



plt.plot(range(1, 15), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)

plt.axis([-1, 16, -1000, 9000000])
plt.show()
