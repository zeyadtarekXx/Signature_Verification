
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import random
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import gc
import pickle

SIZE =224

def read_files(str):
    data = []
    for per in os.listdir('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset'):
        count = 0
        for img_path in glob.glob('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\' + per + '\\' + str + '\\*.png'):

            count += 1
            if (count > 40 and str == 'Train') or (count > 8 and str == 'Test'):
                break

            img_name = img_path.split('\\')[-1]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (224, 224))

            data.append([img, per, img_name])

    random.shuffle(data)
    return data


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


def cluster(desc_stack, n_cluster):
    kmeans_model = KMeans(n_clusters=n_cluster)
    cluster_model = kmeans_model.fit(desc_stack)

    return cluster_model


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


def merge_labels(features, t, per):
    csv_path = 'F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\' + per + '\\' + t + '\\' + per + '_SigVerification' + t + 'Labels.csv'

    per_csv = pd.read_csv(csv_path)

    per_groups = features.groupby(['person_name'])

    per_data = per_groups.get_group(per)

    labeled_data = per_data.merge(per_csv, how='inner', on='image_name')

    enc = OrdinalEncoder()
    enc_labels = enc.fit_transform(np.array(labeled_data['label']).reshape(-1, 1))
    labeled_data['enc_label'] = pd.DataFrame(enc_labels)

    labeled_data.drop(['image_name', 'person_name', 'label'], axis=1, inplace=True)

    return labeled_data


def RUN(n_cluster):
    models = []
    for per in os.listdir('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset'):

        #train
        files = read_files("Train")

        desc_stack = sift(files)

        cluster_model = cluster(desc_stack, n_cluster)

        Features = feature_exctract(cluster_model, files, desc_stack, n_cluster)

        Data = merge_labels(Features, "Train", per)

        X_train, Y_train = Data.iloc[:, :-1], Data["enc_label"]

        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,), random_state=1)

        clf.fit(X_train, Y_train)

        # test
        files_test = read_files("Test")

        desc_stack_test = sift(files_test)

        Features_test = feature_exctract(cluster_model, files_test, desc_stack_test, n_cluster)

        Data_test = merge_labels(Features_test, "Test", per)

        X_test, Y_test = Data_test.iloc[:, :-1], Data_test["enc_label"]

        accuracy = clf.score(X_test, Y_test)

        print("Accuracy:", accuracy)

    return

RUN(50)


