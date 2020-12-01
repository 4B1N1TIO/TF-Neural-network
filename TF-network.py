###############################################
# Tensorflow Neural-Network for OCR numbers
# Author:       Adrian Thierbach
# Date:         28.11.2020
# Dependencies: Python3,numpy,scipy,tensorflow
###############################################

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# load mnist datasets from KERAS

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# scale the image data points to a range of [0,1]

train_images = train_images / 255.0

test_images = test_images / 255.0

###################################################
# define the neural network
###################################################

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(250, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# train the network with 10 generations

model.fit(train_images, train_labels, epochs=10)


# test the network 

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


