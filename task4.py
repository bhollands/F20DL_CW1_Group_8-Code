import sys
assert sys.version_info >= (3, 5)
# Python ≥3.5 is required

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import tarfile
import urllib
import pandas as pd

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the dataset here


# Task 4
# 1. Filter through the dataset and group all images to the relevant classes
# 2. For each class, record the first 10 pixels in order of the absolute correlation value, for each street sign
# 3. Contain data for use in task 5

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
