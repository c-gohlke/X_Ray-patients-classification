import pandas as pd
import numpy as np
import imageio
import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator


import cv2

def load_images():
     print("Loading the images")

     df = pd.read_csv('MURA-v1.1/train_labeled_studies2.csv')
     paths = df["PATH"]

     process = len(paths) #replaced because of memory errors

     #make np array directly (faster)
     X = np.zeros((len(paths), 512, 512))
     # y = np.zeros(len(paths))
     y = np.zeros(process)

     dim = (512, 512)

     for i in range (process):
          if("XR_HAND" in paths[i]):#only load pics with hands for now
               for im_path in glob.glob(paths[i]+"*.png"):
                    im = load_image(im_path)
                    X[i] = np.array(cv2.resize(im, dim))
                    y[i] = (df["DIAGNOSIS"][i])
	 
     print("images loaded")
     return X,y