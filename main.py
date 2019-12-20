import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt

tf.autograph.set_verbosity(0) #higher number, more logs

def load_data():
    df = pd.read_csv('MURA-v1.1/train_labeled_studies.csv')
    df['DIAGNOSIS'] = df['DIAGNOSIS'].astype(str)

    X_y = [[]]
    for i in range(len(df)):
        dirList = os.listdir(df["PATH"][i])
        for dir in dirList:
            X_y.append([df["PATH"][i] + dir, df["DIAGNOSIS"][i]])
    X_y.pop(0) #X_y's first attribute initialized to be [None,None]

    return pd.DataFrame(X_y, columns = ["PATH", "DIAGNOSIS"])

def create_model():
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, (5,5), activation='relu', input_shape=(512, 512, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax')) # 2 because we have 2 classes (healthy vs unhealthy)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

data = load_data()

#image values are initially from 0 to 255
datagen = ImageDataGenerator(rescale=1./255)
train_df, test_df = train_test_split(data,test_size = 0.2)
traingenerator = datagen.flow_from_dataframe( train_df , directory = None, x_col = 'PATH' , y_col = 'DIAGNOSIS', target_size = (512,512) ,class_mode='categorical', batch_size = 32)
testgenerator = datagen.flow_from_dataframe( test_df , directory = None, x_col = 'PATH' , y_col = 'DIAGNOSIS', target_size = (512,512) ,class_mode='categorical', batch_size = 32)

model = create_model()
history = model.fit_generator(
    traingenerator, 
    epochs=5,
    validation_data = testgenerator,
    validation_steps= len(testgenerator),
    steps_per_epoch = len(traingenerator)
)

#Loss
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

#Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()