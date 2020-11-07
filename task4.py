import sys
assert sys.version_info >= (3, 5)
# Python ≥3.5 is required

# Scikit-Learn ≥0.20 is required
# import sklearn
# assert sklearn.__version__ >= "0.20"

# # Common imports
import numpy as np
# import os
# import tarfile
# import urllib
# import pandas as pd
import csv
import math

# To plot pretty figures
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# Import the data here
img_data = "x_train_gr_smpl_random_reduced.csv"
lbl_data = "y_train_smpl_random.csv"
# file = "test.csv"

# Task 4
# 1. Filter through the dataset and group all images to the relevant classes
# 2. For each class, record the first 10 pixels in order of the absolute correlation value, for each street sign
# 3. Contain data for use in task 5

# Step 1: Index of image in x_train corresponds to index of its class label in y_train, so:
#         - Locate and store the positions of each class label in y_train in separate arrays
#         - Filter through x_train data to locate and store the correct images for each class into their class array, using the positions

# Step 2: For each class, find and record the top 10 pixels of each image, in order of absolute correlation value

l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []

f = open(img_data, "r", encoding='utf-8')
imgs = csv.reader(f, quotechar='"')
imgs = list(imgs)

# Fixing problem where first image contains some float values, convert to int
imgs[0] = [float(x) for x in imgs[0]]
imgs[0] = [int(x) for x in imgs[0]]

# Convert all the image data from string to int
for i in range(9690):
    imgs[i] = [int(x) for x in imgs[i]]

f = open(lbl_data, "r", encoding='utf-8')
labels = csv.reader(f, quotechar='"')

i = 0
for label in labels:
    if label[0] == '0':
        l0.append(imgs[i])
    if label[0] == '1':
        l1.append(imgs[i])
    if label[0] == '2':
        l2.append(imgs[i])
    if label[0] == '3':
        l3.append(imgs[i])
    if label[0] == '4':
        l4.append(imgs[i])
    if label[0] == '5':
        l5.append(imgs[i])
    if label[0] == '6':
        l6.append(imgs[i])
    if label[0] == '7':
        l7.append(imgs[i])
    if label[0] == '8':
        l8.append(imgs[i])
    if label[0] == '9':
        l9.append(imgs[i])
    i+=1

class0 = np.array(l0)
class1 = np.array(l1)
class2 = np.array(l2)
class3 = np.array(l3)
class4 = np.array(l4)
class5 = np.array(l5)
class6 = np.array(l6)
class7 = np.array(l7)
class8 = np.array(l8)
class9 = np.array(l9)

print(class0[0])

# For each image in each class, find the top 10 pixels correlating with the class




# feature = classx[0][i]
# corr = pearsonr(list(feature),list(x))
# corrs0.append((float(feature), corr))

# corrs0 = sorted(corrs0, key=float)
# corrs0 = corrs0[:10]