import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn import metrics
from skimage import color
import imageio
import glob
import cv2
import matplotlib.pyplot as plt

############ EXAMPLE PART

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# # Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
# print("Image Data Shape" , digits.data.shape)
# # Print to show there are 1797 labels (integers from 0–9)
# print("Label Data Shape", digits.target.shape)

# logreg = LogisticRegression(C=1e20, solver='liblinear', max_iter=200, multi_class='auto')
# logreg.fit(X_train, y_train)
# score = logreg.score(X_test, y_test)
# print(score)

########### MURA PART

def load_images():
     print("Loading the images")

     df = pd.read_csv('MURA-v1.1/train_labeled_studies2.csv')
     paths = df["PATH"]

     process = len(paths) #replaced because of memory errors

     #make np array directly (faster)
     # X = np.zeros((len(paths), 262144))
     X = np.zeros((process, 262144))
     # y = np.zeros(len(paths))
     y = np.zeros(process)

     dim = (512, 512) #original is 512*512
     #for each path in the .csv
     # for i in range (len(paths)):
     for i in range (process):
          if("XR_HAND" in paths[i]):#only load pics with hands for now
               for im_path in glob.glob(paths[i]+"*.png"):
                    im = color.rgb2gray(imageio.imread(im_path, as_gray=True))
                    res = cv2.resize(im, dim)
                    x = np.array(res)
                    # x = x.reshape(-1)

                    X[i] = x
                    y[i] = (df["DIAGNOSIS"][i])
     
     print("images loaded")
     return X,y

def display_one(a):
    plt.imshow(a)
    plt.show()

images,labels = load_images()
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=0)

unique, counts = np.unique(y_train, return_counts=True)
print("y_train distribution is", unique, counts)

print("Image Data Shape" , images.shape)
print("Label Data Shape", labels.shape)

logreg = LogisticRegression(C=1e10, solver='lbfgs')
logreg.fit(X_train, y_train)

print("model done training")

y_hat = logreg.predict(X_train)
train_acc = metrics.accuracy_score(y_hat, y_train)
print("training accuracy is ", train_acc)
#logreg.predict_proba(X_train)
y_hat_test = logreg.predict(X_test)
test_acc = metrics.accuracy_score(y_hat_test, y_test)
print("testing accuracy is ", test_acc)

unique, counts = np.unique(y_hat_test, return_counts=True)
print("y_hat_test distribution is ", unique, counts)
unique, counts = np.unique(y_test, return_counts=True)
print("y_test distribution is ", unique, counts)