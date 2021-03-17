#!/usr/bin/env python
# coding: utf-8
#Bee vs Wasp Classification
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pathlib
import random

dataset_path = 'C:/Users/user/beeVSwasp'
os.listdir(dataset_path)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    #image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [128, 128])
    image = image / 255
    return image

def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    #image = preprocess_image(image)
    return preprocess_image(image)

#showing image with bee
bee_img =  mpimg.imread(dataset_path+'/bee/22874935_066a5b774c_m.jpg')
plt.imshow(bee_img)
plt.title("Bee")

#show image with wasp
wasp_img =  mpimg.imread(dataset_path+'/wasp/88711747_76c3efdcaa_m.jpg')
plt.imshow(wasp_img)
plt.title("Wasp")

pathlib_dataset = pathlib.Path(dataset_path)
all_image_paths = list(str(x) for x in pathlib_dataset.glob('*/*'))
#random.shuffle(all_image_paths)
DATASET_SIZE = len(all_image_paths)
print(DATASET_SIZE)

#preparing dataset as numpy array
Y = np.asarray([[pathlib.Path(path).parent.name] for path in all_image_paths])
Y = np.where(Y=='bee', 1, Y)
Y = np.where(Y=='wasp', 0, Y)
Y = np.asarray(Y, dtype=np.float32)
X = np.asarray(load_and_preprocess_image(all_image_paths[0]))
X = np.expand_dims(X, axis=0)
for img_path in all_image_paths[1:]:
    image = np.expand_dims(np.asarray(load_and_preprocess_image(img_path)), axis=0)
    X = np.append(X, image, axis=0)
#splitting dataset into train and data
(trainX, testX, trainY, testY) = train_test_split(X,
    Y, test_size=0.25, random_state=42)

#building conv NN
def conv_net(X, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.compat.v1.variable_scope('ConvNet', reuse=reuse):
        x = X['images']
        # Convolution Layer with 64 filters and a kernel size of 5
        conv1 = tf.compat.v1.layers.conv2d(x, 64, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.compat.v1.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 3
        conv2 = tf.compat.v1.layers.conv2d(conv1, 32, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.compat.v1.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.compat.v1.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.compat.v1.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.compat.v1.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.compat.v1.layers.dense(fc1, 1)

    return out

learning_rate = 0.001
num_steps = 1000
batch_size = 64

# Network Parameters
dropout = 0.25

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Building the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = (logits_test > 0.5)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_train, labels=labels))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.compat.v1.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

model = tf.estimator.Estimator(model_fn)

input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'images': trainX}, y=trainY,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'images': testX}, y=testY,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
eval = model.evaluate(input_fn)

print("Testing Accuracy:", eval['accuracy'])
