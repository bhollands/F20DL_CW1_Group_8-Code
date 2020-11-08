import sys
assert sys.version_info >= (3, 5)
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

val = int(input("Which file do you want classified? 1 for Q3, 2 for Q5.1, 3 for Q5.2 and 4 for Q5.3: "))
image_data = ""

if val == 1:
    image_data = "x_train_gr_smpl_random_reduced.csv"
    
elif val == 2:
    image_data = "train_sampl_1.csv"
    
elif val == 3:
    image_data = "train_sampl_2.csv"
elif  val == 4:
    image_data = "train_sampl_3.csv"
    
print(val)
print(image_data)

label_data = "y_train_smpl_random.csv"

#reads the data from the files
x_data = pd.read_csv(image_data)
y_data = pd.read_csv(label_data)

df1 = pd.DataFrame(x_data)
df2 = pd.DataFrame(y_data)

#turns the dataframes to numpy arrays
X= df1.values
#ravel flattens the labels array into a one dimensional array
Y= df2.values.ravel()

# Preprocessing to check if the datatypes and arrays are ready to use
print(X.dtype, Y.dtype)
print(X.shape, Y.shape)

# Trains the model for 70% of the images array and 30% of the labels
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y)

train_X.shape, test_X.shape

#Using the multinomial baive bayes algorithm as it is suitable for classification with discrete features i.e. the pixels that make up the images
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
clas = MultinomialNB()
clas.fit(train_X, train_Y)

#Returns the accuracy on the given test data and labels.
clas.score(test_X, test_Y)

from sklearn.metrics import classification_report
predictions = clas.predict(test_X)

#Test used in Q5 to check if every label has been predicted
labels1 = set(test_Y) - set(predictions)

for i in labels1:
    print(i)
    
print(classification_report(test_Y, predictions, labels=np.unique(predictions)))

def plot_images(images, labels):
    n_cols = min(5, len(images))
    n_rows = len(images) //n_cols
    fig = plt.figure(figsize=(8, 8))
    
    for i in range(n_rows * n_cols):
        sp = fig.add_subplot(n_rows, n_cols, i+1)
        plt.axis("off")
        plt.imshow(images[i], cmap=plt.cm.gray)
        sp.set_title(labels[i])
plt.show()

p = np.random.permutation(len(test_X))
p = p[:20]

dataset = image_data
if dataset == 'x_train_gr_smpl_random_reduced.csv':
    plot_images(test_X[p].reshape(-1, 35, 35), predictions[p])

elif dataset == 'train_sampl_1.csv':
    plot_images(test_X[p].reshape(-1, 2, 2), predictions[p])

elif dataset == 'train_sampl_2.csv':
    plot_images(test_X[p].reshape(-1, 5, 5), predictions[p])

elif  dataset == 'train_sampl_3.csv':
    plot_images(test_X[p].reshape(-1, 4, 4), predictions[p])

ycount = np.ones((10))

#Calculates the probablities of each class
for x, y in zip(X, Y):
    y1 = int(y)
    ycount[y1] += 1
        
py = (ycount/ ycount.sum()) * 100

print('Probablities', py)

classifier = clas

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, test_X, test_Y,
                                 display_labels=Y,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

