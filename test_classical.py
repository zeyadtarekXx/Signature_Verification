import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn import metrics
import cv2
import os
import glob
import matplotlib.pyplot as plt
import random


def read_files(str):
    data = []
    for per in os.listdir('F:\COLLEGE\Computer Vision\PROJECT\CV_2023_SC_Dataset'):
        count = 0
        for img_path in glob.glob('F:\COLLEGE\Computer Vision\PROJECT\CV_2023_SC_Dataset' + per + '/' + str + '/*.png'):

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
    csv_path = 'F:\COLLEGE\Computer Vision\PROJECT\CV_2023_SC_Dataset' + per + '/' + t + '/' + per + '_SigVerification' + t + 'Labels.csv'

    per_csv = pd.read_csv(csv_path)

    per_groups = features.groupby(['person_name'])

    per_data = per_groups.get_group(per)

    labeled_data = per_data.merge(per_csv, how='inner', on='image_name')

    enc = OrdinalEncoder()
    enc_labels = enc.fit_transform(np.array(labeled_data['label']).reshape(-1, 1))
    labeled_data['enc_label'] = pd.DataFrame(enc_labels)

    labeled_data.drop(['image_name', 'person_name', 'label'], axis=1, inplace=True)

    return labeled_data


def train_model(n_cluster):
    models = []
    for per in os.listdir('F:\COLLEGE\Computer Vision\PROJECT\CV_2023_SC_Dataset'):
        files = read_files("Train")

        desc_stack = sift(files)

        cluster_model = cluster(desc_stack, n_cluster)

        Features = feature_exctract(cluster_model, files, desc_stack, n_cluster)

        Data = merge_labels(Features, "Train", per)

        X_train, Y_train = Data.iloc[:, :-1], Data["enc_label"]

        classifier = SVC(C=1.0, kernel='linear')

        classifier.fit(X_train, Y_train)

        models.append(classifier)

    return models, cluster_model


def test_model(cluster_model, models, n_cluster):
    count = 0

    for per in os.listdir('F:\COLLEGE\Computer Vision\PROJECT\CV_2023_SC_Dataset'):
        files = read_files("Test")

        desc_stack = sift(files)

        Features = feature_exctract(cluster_model, files, desc_stack, n_cluster)

        Data = merge_labels(Features, "Test", per)

        X_test, Y_test = Data.iloc[:, :-1], Data["enc_label"]

        y_predict = models[count].predict(X_test)

        print("Accuracy score %.3f" % metrics.accuracy_score(Y_test, y_predict))

        count += 1



models, cluster_model = train_model(50)
test_model(cluster_model, models, 50)