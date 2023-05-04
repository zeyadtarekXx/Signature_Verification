import random
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob

def read_files(str):
    data = []
    for per in os.listdir('F:\COLLEGE\\Computer Vision\\PROJECT\\SignatureObjectDetection'):
        for img_path in glob.glob('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\' + per + '\\' + str + '\\*.png'):

            img_name = img_path.split('.')[0]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (224, 224))

            data.append([img, per, img_name])

    random.shuffle(data)
    return data
