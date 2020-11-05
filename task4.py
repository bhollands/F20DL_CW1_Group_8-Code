import sys
assert sys.version_info >= (3, 5)
# Python ≥3.5 is required

# Scikit-Learn ≥0.20 is required
# import sklearn
# assert sklearn.__version__ >= "0.20"

# # Common imports
# import numpy as np
# import os
# import tarfile
# import urllib
# import pandas as pd
import csv

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

# Step 2: For each class, find and record the top 10 pixels in order of absolute correlation value

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
img_list = list(imgs)

f = open(lbl_data, "r", encoding='utf-8')
labels = csv.reader(f, quotechar='"')

i = 0
for label in labels:
    # if i == 9690:
        # break
    if label[0] == '0':
        l0.append(img_list[i])
        print(i)
    if label[0] == '1':
        l1.append(img_list[i])
        print(i)
    if label[0] == '2':
        l2.append(img_list[i])
        print(i)
    if label[0] == '3':
        l3.append(img_list[i])
        print(i)
    if label[0] == '4':
        l4.append(img_list[i])
        print(i)
    if label[0] == '5':
        l5.append(img_list[i])
        print(i)
    if label[0] == '6':
        l6.append(img_list[i])
        print(i)
    if label[0] == '7':
        l7.append(img_list[i])
        print(i)
    if label[0] == '8':
        l8.append(img_list[i])
        print(i)
    if label[0] == '9':
        l9.append(img_list[i])
        print(i)
    i+=1


# f = open(file, "r", encoding='utf-8')
# reader = csv.reader(f, quotechar='"')

# img_list = list(reader)
# print(img_list[0])

# img_list = list(lbl_reader)
# print(int(img_list[0][0]))


# for row in reader:
#     print(row)

# 9690/1(?) images - 10 classes (0-9)

# Class 0 - 211 images

# Class 1 - 2220 images

# Class 2 - 2250 images

# Class 3 - 1410 images

# Class 4 - 1980 images

# Class 5 - 210 images

# Class 6 - 360 images

# Class 7 - 240 images

# Class 8 - 540 images

# Class 9 - 270 images