import numpy as np
import os
import pandas as pd
import os
import cv2
import glob
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.applications import VGG19, VGG16, ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Activation,Dropout
from keras.utils import to_categorical
import tensorflow as tf
import random
from keras.models import Sequential, Model, load_model



train_data_names = []
test_data_names = []

train_data = []
train_labels = []

train_data_names = []
test_data_names = []

train_data = []
train_labels = []

SIZE = 224


def create_label(image_name):
    label = image_name.split("\\")[-3]
    if label == "personA":
        return np.array([1, 0, 0, 0, 0])
    elif label == "personB":
        return np.array([0, 1, 0, 0, 0])
    elif label == "personC":
        return np.array([0, 0, 1, 0, 0])
    elif label == "personD":
        return np.array([0, 0, 0, 1, 0])
    elif label == "personE":
        return np.array([0, 0, 0, 0, 1])



for per in os.listdir('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset'):
    count = 0
    for data in glob.glob('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\' + per + '\\' + 'Train' + '\\*.png'):

        count += 1
        if (count > 40):
            break

        train_data_names.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        train_data.append([img])
        train_labels.append(create_label(data))



train_data = np.array(train_data) / 255.0
train_labels = np.array(train_labels)

# Test Data

test_data = []
test_labels = []

for per in os.listdir('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset'):
    count = 0
    for data in glob.glob(
            'F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\' + per + '\\' + 'Test' + '\\*.png'):

        count += 1
        if (count > 8):
            break

        test_data_names.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        test_data.append([img])
        test_labels.append(create_label(data))





test_data = np.array(test_data) / 255.0
test_labels = np.array(test_labels)


#Train-validation-test split
x_train_old,x_val_old,y_train_old,y_val_old=train_test_split(train_data,train_labels,test_size=.3)

y_train = np.expand_dims(y_train_old,-1)
y_val = np.expand_dims(y_val_old,-1)
test_labels = np.expand_dims(test_labels,-1)

x_train = np.squeeze(x_train_old, axis=1)
x_val = np.squeeze(x_val_old, axis=1)
test_data = np.squeeze(test_data, axis=1)



print((x_train.shape,y_train.shape))
print((x_val.shape,y_val.shape))
print((test_data.shape,test_labels.shape))

# y_train=to_categorical(y_train)
# y_val=to_categorical(y_val)
# y_test=to_categorical(test_labels)

print((x_train.shape,y_train.shape))
print((x_val.shape,y_val.shape))
print((test_data.shape,test_labels.shape))

#Image Data Augmentation
train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1)

test_generator = ImageDataGenerator(rotation_range=2,  horizontal_flip= True, zoom_range=.1)

#Fitting the augmentation defined above to the data
train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(test_data)

#Learning Rate Annealer
lrr= ReduceLROnPlateau(monitor='loss', factor=.01,  patience=3, min_lr=1e-5)

base_model_VGG19 = VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=5)


model_vgg19 = Sequential()
model_vgg19.add(base_model_VGG19)
model_vgg19.add(Flatten())
model_vgg19.add(Dense(1024,activation=('relu'),input_dim=512))
model_vgg19.add(Dense(512,activation=('relu')))
model_vgg19.add(Dense(256,activation=('relu')))
model_vgg19.add(Dropout(.3))
model_vgg19.add(Dense(128,activation=('relu')))
model_vgg19.add(Dropout(.5))
model_vgg19.add(Dense(5,activation=('softmax')))



#VGG19 Model Summary
model_vgg19.summary()

#Defining the hyperparameters
batch_size= 100
epochs=30
learn_rate=.001
sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)

#Compiling the VGG19 model
model_vgg19.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])


model_vgg19.fit_generator(train_generator.flow(x_train, y_train, batch_size = batch_size), epochs=epochs, steps_per_epoch = x_train.shape[0]//batch_size, validation_data = val_generator.flow(x_val, y_val, batch_size = batch_size), validation_steps = 250, callbacks = [lrr], verbose = 1)

model_vgg19.save('model_vgg19_off.h5')
model_vgg19.save_weights('model_vgg19_weights_off.h5')

# model_vgg19 = load_model('model_vgg19.h5')

#Making prediction


# y_pred1 = np.argmax(model_vgg19.predict(test_data), axis=1)
# y_true = np.argmax(y_test,axis=1)
#
# from sklearn.metrics import accuracy_score
# accuracy_score(y_true, y_pred1)
#Plotting the confusion matrix
# confusion_mtx=confusion_matrix(y_true,y_pred1)
#
# class_names=['Person A' , 'Person B' , 'Person C' , 'Person D' , 'Person E']
#
# #Plotting non-normalized confusion matrix
# plot_confusion_matrix(y_true, y_pred1, classes = class_names,  title = 'Non-Normalized VGG19 Confusion Matrix')
