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

# common imports
import numpy as np 
import os
import cv2
import pandas as pd
#to make thos notebooks output stable across runs
np.random.seed(42)

#to plot pretty sigures
import matplotlib as mpl
import matplotlib.pyplot as plt

#Import all the data
here = os.path.dirname(os.path.abspath(__file__))
x_train_smpl_bin_random  = os.path.join(here, 'x_train_smpl_bin_random.csv')
x_train_smpl_bin_random_reduced = os.path.join(here, 'x_train_smpl_bin_random_reduced.csv')

x_train_gr_smpl_random = os.path.join(here, 'x_train_gr_smpl_random.csv')
x_train_gr_smpl_random_reduced = os.path.join(here, 'x_train_gr_smpl_random_reduced.csv')