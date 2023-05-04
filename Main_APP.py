import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import applications
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, \
    TensorBoard
from tensorflow.keras import backend as K
import gc
from tensorflow.keras.models import Model
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Computing confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

# Visualizing
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

   # Rotating the tick labels and setting their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Looping over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

train_data_names = []
test_data_names = []

train_data = []
train_labels = []

train_data_names = []
test_data_names = []

train_data = []
train_labels = []

SIZE = 224

for per in os.listdir('F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset'):
    count = 0
    for data in glob.glob(
            'F:\\COLLEGE\\Computer Vision\\PROJECT\\CV_2023_SC_Dataset\\' + per + '\\' + 'Train' + '\\*.png'):

        count += 1
        if (count > 40):
            break

        train_data_names.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        train_data.append([img])
        if per[-1] == 'g':
            train_labels.append(np.array(1))
        else:
            train_labels.append(np.array(0))

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
        if per[-1] == 'g':
            test_labels.append(np.array(1))
        else:
            test_labels.append(np.array(0))

test_data = np.array(test_data) / 255.0
test_labels = np.array(test_labels)

with open('./train_data_names.pkl', 'wb') as fp:
    pickle.dump(train_data_names, fp)

with open('./test_data_names.pkl', 'wb') as fp:
    pickle.dump(test_data_names, fp)

# Categorical labels
# print(train_labels)
train_labels = to_categorical(train_labels)
# print(train_data.shape)
# Reshaping
train_data = train_data.reshape(-1, SIZE, SIZE, 3)
test_data = test_data.reshape(-1, SIZE, SIZE, 3)

input_ = (224, 224, 3)
EPOCHS = 2
BS = 64
output_ = 5

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_)

model = Sequential()
data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.1)])
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(output_, activation='softmax'))

model = Model(inputs=model.input, outputs=model.output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.summary()

earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1)

early_stop = [earlyStopping]
progess = model.fit(train_data, train_labels, batch_size=BS, epochs=EPOCHS, callbacks=early_stop, validation_split=.3)
# progess = model.fit_generator(train_data, train_labels, epochs = EPOCHS ,callbacks=early_stop, validation_split=.3)


acc = progess.history['accuracy']
val_acc = progess.history['val_accuracy']
loss = progess.history['loss']
val_loss = progess.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-2].output)

intermediate_output_train = intermediate_layer_model.predict(train_data)
intermediate_output_test = intermediate_layer_model.predict(test_data)



np.save('./VGG16_Adam_train_5', intermediate_output_train)
np.save('./VGG16_Adam_test_5', intermediate_output_test)

model.save_weights("model_vgg_base_5_weights_non.h5")
model.save_weights("model_vgg_base_5_non.h5")


# model_vgg = tf.keras.models.load_model('model_vgg_base_5.h5')

# print(f"Macro-averaged Precision score : {macro_precision(test_labels, intermediate_output_test) }")
#test accuracy

# #Making prediction
# y_pred2=model_vgg.predict(test_data)
# y_true=np.argmax(test_labels,axis=0)
#
# #Plotting the confusion matrix
# confusion_mtx=confusion_matrix(y_true,y_pred2)
#
# class_names=['person A', 'person B', 'person C', 'person D', 'person E']
#
# #Plotting non-normalized confusion matrix
# plot_confusion_matrix(y_true, y_pred2, classes = class_names,title = 'Non-Normalized VGG16 Confusion Matrix')
#
# from sklearn.metrics import accuracy_score
# accuracy_score(y_true, y_pred2)