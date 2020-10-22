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

here = os.path.dirname(os.path.abspath(__file__))
x_train_smpl_bin_random  = os.path.join(here, 'x_train_smpl_bin_random.csv')
x_train_smpl_bin_random_reduced = os.path.join(here, 'x_train_smpl_bin_random_reduced.csv')

x_train_gr_smpl_random = os.path.join(here, 'x_train_gr_smpl_random.csv')
x_train_gr_smpl_random_reduced = os.path.join(here, 'x_train_gr_smpl_random_reduced.csv')


data = pd.read_csv(x_train_smpl_bin_random)
#data = pd.read_csv(x_train_gr_smpl_random)
resize_length = 35

num_of_rows = 2431#9691#2431
num_of_colums_original = 2304
num_of_colums_reduced = resize_length**2


row = np.empty(num_of_colums_original) #row needs to be as long as the number of columbs

reduced_data_set = np.zeros(shape=(num_of_rows,num_of_colums_reduced)) #2D array to store entire set


for j in range(num_of_rows): #replace number with num_of_rows
    print(j)
    for i in range(num_of_colums_original):
        row[i] = data[str(i)][j] #have this array, need to put into 2D according to larger for loop

    #plt.imshow(row, cmap=mpl.cm.binary)
    #plt.show()
    image = cv2.resize(row.reshape(48,48), dsize=(resize_length,resize_length)) #re-size the images

    reduced_image_vector = image.reshape(1,num_of_colums_reduced) #set the reduced image to a vector again
    #plt.imshow(image, cmap=mpl.cm.binary)
    #plt.show()
    
 
    reduced_data_set[j] = reduced_image_vector #write the reduced image to the 2d array

#print(entire_data_set)
#normalized_data = entire_data_set/255
#print(normalized_data)

np.savetxt(x_train_smpl_bin_random_reduced, reduced_data_set, delimiter=',', fmt='%d')
#np.savetxt(x_train_gr_smpl_random_reduced, reduced_data_set, delimiter=',', fmt='%d')

#DF = pd.DataFrame(normalized_data)
#DF.to_csv(reduced_data_csv)


#row_image = row.reshape(48,48)
#plt.imshow(row_image, cmap=mpl.cm.binary)
