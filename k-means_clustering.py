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