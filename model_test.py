#!/bin/python3

# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.client import device_lib
from sklearn.metrics import classification_report

cls_list = ['normal', 'flooding', 'injection', 'impersonation']
bin_list = ['normal', 'attack']

conf_dict = {
    "Machine": "COLAB PRO",
    "CPU": "None",
    "batch-size": 200,
    "epochs": 15,
    "learning_rate": 0.00000025,
    "GPU" : "Not present"
}

ENDING = '_36b_minmax_red_final'

print('GPU name: ', device_lib.list_local_devices())
conf_dict['GPU'] = device_lib.list_local_devices()

print(tf.__version__)

x_train = pd.read_csv("test_set{}.csv".format(ENDING))
y_train = x_train.loc[:,'class']

x_test = pd.read_csv("test_set{}.csv".format(ENDING))
y_test = x_test.loc[:,'class']


print(x_train.info())
print(x_test.info())

num_classes = 4
test_classes = 4

try:
    x_train.drop(columns=['class'], inplace=True)
    x_test.drop(columns=['class'], inplace=True)
except Exception as e:
    print(e)

x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')

x_test = np.array(x_test).astype('float32')
y_test = np.array(y_test).astype('float32')

print(x_train.shape)
print(x_test.shape)

x_train = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, test_classes)


print(f"X TRAIN SHAPE: {x_train.shape}")
print(f"Num classes: {num_classes}")


adam = tf.keras.optimizers.Adam(learning_rate=conf_dict['learning_rate'])
model = tf.keras.models.load_model('.cnn_cls_94')

model.summary()

model.compile(optimizer=adam,
                loss="categorical_crossentropy",
                metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
pr = model.predict(x_train, verbose=1, batch_size=200)
y_pred = np.argmax(pr, axis=1)
y_train = np.argmax(y_train, axis=1)

res = classification_report(y_pred, y_train, digits=15)

print(res)