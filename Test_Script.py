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
import pickle
#load models
model = load_model('model_vgg19.h5')
model_2 = load_model('model_vgg19.h5')

#preprocess
SIZE = 224
img_path = 'F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\personE\\Test\\personE_17.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (SIZE, SIZE))

#stage 1
classify = model.predict(img)

if classify == "Person A":
    data = pickle.load('F:\\COLLEGE\\Computer Vision\\PIck\\test_data_personA2.pkl')
elif classify == "Person B":
    data = pickle.load('F:\\COLLEGE\\Computer Vision\\PIck\\test_data_personB2.pkl')
elif classify == "Person C":
    data = pickle.load('F:\\COLLEGE\\Computer Vision\\PIck\\test_data_personC2.pkl')
elif classify == "Person D":
    data = pickle.load('F:\\COLLEGE\\Computer Vision\\PIck\\test_data_personD2.pkl')
elif classify == "Person E":
    data = pickle.load('F:\\COLLEGE\\Computer Vision\\PIck\\test_data_personE2.pkl')



#stage 2
real_fake = model.predict(classify)





