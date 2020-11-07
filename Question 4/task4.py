import sys
assert sys.version_info >= (3, 5)
# Python ≥3.5 is required

# Scikit-Learn ≥0.20 is required
# import sklearn
# assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
# import os
# import tarfile
# import urllib
# import pandas as pd
import pandas as pd
import csv
import math
from scipy.stats import pearsonr
  
# To plot pretty figures
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# Import the data here
img_data = "x_train_gr_smpl_random_reduced.csv"
lbl_data = "y_train_smpl_random.csv"

f = open(lbl_data, "r", encoding='utf-8')
labels = csv.reader(f, quotechar='"')
labels = list(labels)

f = open(img_data, "r", encoding='utf-8')
imgs = csv.reader(f, quotechar='"')
imgs = list(imgs)

# Fixing problem where first image contains some float values, convert to int and back to string
imgs[0] = [float(x) for x in imgs[0]]
imgs[0] = [int(x) for x in imgs[0]]
imgs[0] = [str(x) for x in imgs[0]]
  
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

# Class 0 - 714,542,507,749,577,680,783,784,748,545,580,822,818,715,19,679,54,471,615,55
# Class 1 - 542,646,541,752,577,576,717,611,786,506,682,793,681,751,645,545,819,792,783,510
# Class 2 - 319,284,285,320,249,318,283,248,353,247,354,214,250,317,421,352,386,286,282,213
# Class 3 - 644,678,679,713,645,643,714,748,609,610,646,680,712,677,783,749,753,829,752,682
# Class 4 - 542,1140,1141,543,1109,1110,1175,1142,1111,1096,1201,1108,1143,1200,507,1146,1095,1097,1145,1144
# Class 5 - 677,642,861,826,406,371,683,719,370,643,684,401,405,570,791,400,608,756,535,726
# Class 6 - 439,440,404,405,403,618,582,617,583,438,678,402,653,441,677,547,401,643,652,856
# Class 7 - 648,583,642,405,607,613,406,618,548,441,573,676,369,677,619,440,649,726,370,547
# Class 8 - 406,642,405,677,583,607,370,548,441,676,711,573,619,572,400,641,369,726,608,401
# Class 9 - 642,677,607,618,860,583,676,824,573,572,582,641,401,825,440,547,608,711,859,653

# Top 5 features per class  
i = 0
for label in labels:
    if label[0] == '0':
        l0.append([imgs[i][714],imgs[i][542],imgs[i][507],imgs[i][749],imgs[i][577]])

    if label[0] == '1':
        l1.append([imgs[i][542],imgs[i][646],imgs[i][541],imgs[i][752],imgs[i][577]])

    if label[0] == '2':
        l2.append([imgs[i][319],imgs[i][284],imgs[i][285],imgs[i][320],imgs[i][249]])

    if label[0] == '3':
        l3.append([imgs[i][644],imgs[i][678],imgs[i][679],imgs[i][713],imgs[i][645]])

    if label[0] == '4':
        l4.append([imgs[i][542],imgs[i][1140],imgs[i][1141],imgs[i][543],imgs[i][1109]])

    if label[0] == '5':
        l5.append([imgs[i][677],imgs[i][642],imgs[i][861],imgs[i][826],imgs[i][406]])

    if label[0] == '6':
        l6.append([imgs[i][439],imgs[i][440],imgs[i][404],imgs[i][405],imgs[i][403]])

    if label[0] == '7':
        l7.append([imgs[i][648],imgs[i][583],imgs[i][642],imgs[i][405],imgs[i][607]])

    if label[0] == '8':
        l8.append([imgs[i][406],imgs[i][642],imgs[i][405],imgs[i][677],imgs[i][583]])

    if label[0] == '9':
        l9.append([imgs[i][642],imgs[i][677],imgs[i][607],imgs[i][618],imgs[i][860]])
    i+=1

dataset1 = l0+l1+l2+l3+l4+l5+l6+l7+l8+l9
columns = [str(x) for x in range(5)]

df1 = pd.DataFrame(dataset1, columns=columns)
df1.to_csv('/Users/Perry/Documents/GitHub/F20DL_CW1_Group_8-Code/Question 5/train_sampl_1.csv', index=False, header=False)

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

# Top 10 features per class
i = 0
for label in labels:
    if label[0] == '0':
        l0.append([imgs[i][714],imgs[i][542],imgs[i][507],imgs[i][749],imgs[i][577],
                    imgs[i][680],imgs[i][783],imgs[i][784],imgs[i][748],imgs[i][545]])

    if label[0] == '1':
        l1.append([imgs[i][542],imgs[i][646],imgs[i][541],imgs[i][752],imgs[i][577],
                    imgs[i][576],imgs[i][717],imgs[i][611],imgs[i][786],imgs[i][506]])

    if label[0] == '2':
        l2.append([imgs[i][319],imgs[i][284],imgs[i][285],imgs[i][320],imgs[i][249],
                    imgs[i][318],imgs[i][283],imgs[i][248],imgs[i][353],imgs[i][247]])

    if label[0] == '3':
        l3.append([imgs[i][644],imgs[i][678],imgs[i][679],imgs[i][713],imgs[i][645],
                    imgs[i][643],imgs[i][714],imgs[i][748],imgs[i][609],imgs[i][610]])
                    
    if label[0] == '4':
        l4.append([imgs[i][542],imgs[i][1140],imgs[i][1141],imgs[i][543],imgs[i][1109],
                    imgs[i][1110],imgs[i][1175],imgs[i][1142],imgs[i][1111],imgs[i][1096]])

    if label[0] == '5':
        l5.append([imgs[i][677],imgs[i][642],imgs[i][861],imgs[i][826],imgs[i][406],
                    imgs[i][371],imgs[i][683],imgs[i][719],imgs[i][370],imgs[i][643]])

    if label[0] == '6':
        l6.append([imgs[i][439],imgs[i][440],imgs[i][404],imgs[i][405],imgs[i][403],
                    imgs[i][618],imgs[i][582],imgs[i][617],imgs[i][583],imgs[i][438]])

    if label[0] == '7':
        l7.append([imgs[i][648],imgs[i][583],imgs[i][642],imgs[i][405],imgs[i][607],
                    imgs[i][613],imgs[i][406],imgs[i][618],imgs[i][548],imgs[i][441]])

    if label[0] == '8':
        l8.append([imgs[i][406],imgs[i][642],imgs[i][405],imgs[i][677],imgs[i][583],
                    imgs[i][607],imgs[i][370],imgs[i][548],imgs[i][441],imgs[i][676]])

    if label[0] == '9':
        l9.append([imgs[i][642],imgs[i][677],imgs[i][607],imgs[i][618],imgs[i][860],
                    imgs[i][583],imgs[i][676],imgs[i][824],imgs[i][573],imgs[i][572]])
    i+=1

print(l0)
dataset2 = l0+l1+l2+l3+l4+l5+l6+l7+l8+l9
columns = [str(x) for x in range(10)]

df2 = pd.DataFrame(dataset2, columns=columns)
print(df2.head())
df2.to_csv('/Users/Perry/Documents/GitHub/F20DL_CW1_Group_8-Code/Question 5/train_sampl_2.csv', index=False, header=False)