import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import applications
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
import keras.utils as image
# from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import os
import pandas as pd
import os
import cv2
import glob
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler


import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#from keras.applications import VGG19, VGG16, ResNet50
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import SGD,Adam

import random
from keras.models import Sequential, Model, load_model
import pickle


def sift(data):
    sift = cv2.SIFT_create()
    desc = []

    for row in data:
        kp, des = sift.detectAndCompute(row[0], None)
        desc.append(des)

    desc_stack = np.array(desc[0])
    for remaining in desc[1:]:
        desc_stack = np.vstack((desc_stack, remaining))

    return desc_stack


def feature_exctract(cluster_model, data, desc_stack, n_cluster):
    clusters = cluster_model.predict(desc_stack)

    histograms = np.array([np.zeros(n_cluster) for i in range(len(data))])

    count = 0
    final_data = []
    for i in range(len(data)):
        l = len(data[i])
        for j in range(l):
            index = clusters[count + j]
            histograms[i][index] += 1
        count += l

    std_histograms = StandardScaler().fit_transform(histograms)

    final_data = []

    for i in range(len(std_histograms)):
        row = []
        for j in range(len(std_histograms[i])):
            row.append(std_histograms[i, j])

        row.append(data[i][1])
        row.append(data[i][-1])

        final_data.append(row)

    columns = []

    for i in range(n_cluster):
        columns.append('feature' + str(i))

    columns.append('person_name')
    columns.append('image_name')

    final_df = pd.DataFrame(final_data, columns=columns)

    return final_df




# preprocess
SIZE = 224
n_clusters = 50
img = cv2.imread(img_path)
img = cv2.resize(img, (SIZE, SIZE))





model = load_model('model_vgg19.h5')

img_path = 'F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\personA\\Test\\personA_2.png'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pre = model.predict(x)
mmm = numpy.argmax(pre , axis=1)+1


classify = ""
if mmm == 1:
    classify = "PersonA"
elif mmm == 2:
    classify = "PersonB"
elif mmm == 3:
    classify = "PersonC"
elif mmm == 4:
    classify = "PersonD"
elif mmm == 5:
    classify = "PersonE"

print(classify)
# stage 1
#
clust_file = open("F:\\COLLEGE\\Computer Vision\\PROJECT\\cluster.pkl", 'rb')
cluster_model = pickle.load(clust_file)

label_file = open("F:\\COLLEGE\\Computer Vision\\PROJECT\\encoder.pkl", 'rb')
label_enc = pickle.load(label_file)
model = Model()

if classify == "PersonA":
    file = open("F:\\COLLEGE\\Computer Vision\\PROJECT\\test_data_personA2.pkl", "rb")
    model = pickle.load(file)


elif classify == "PersonB":
    file = open("F:\\COLLEGE\\Computer Vision\\PROJECT\\test_data_personB2.pkl", "rb")
    model = pickle.load(file)


elif classify == "Person C":
    file = open("F:\\COLLEGE\\Computer Vision\\PROJECT\\test_data_personC2.pkl", "rb")
    model = pickle.load(file)


elif classify == "Person D":
    file = open("F:\\COLLEGE\\Computer Vision\\PROJECT\\test_data_personD2.pkl", "rb")
    model = pickle.load(file)

elif classify == "Person E":
    file = open("F:\\COLLEGE\\Computer Vision\\PROJECT\\test_data_personE2.pkl", "rb")
    model = pickle.load(file)

data = [[np.array(img), classify]]

desc_stack = sift(data)

Features = feature_exctract(cluster_model, data, desc_stack, n_clusters)

y_predict = model.predict(Features.iloc[:, :n_clusters])
print(label_enc.inverse_transform(y_predict.reshape(1, -1)))

# stage 2
# real_fake = data.predict()
# print(real_fake)
