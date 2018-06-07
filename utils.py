import copy
import cv2
import os
import scipy
import numpy as np
import tensorflow as tf

def to_xy(j, k):
    return (k - 13, -j + 14)


def to_jk(x, y):
    return (-y + 14, x + 13)


def image_value(image, x, y):
    j, k = to_jk(x, y)
    return image[int(j)][int(k)][0]

def rotate(image):
    rot_image = copy.deepcopy(image)
    theta = 45 * np.pi / 180

    for j in range(28):
        for k in range(28):
            x, y = to_xy(j, k)
            x1 = np.cos(theta) * x + np.sin(theta) * y
            y1 = -np.sin(theta) * x + np.cos(theta) * y
            x2 = np.floor(x1)
            delta_x = x1 - x2
            y2 = np.floor(y1)
            delta_y = y1 - y2
            if x2 < -13 or x2 > 13 or y2 < -13 or y2 > 13: continue
            value \
                    = (1 - delta_x) * (1 - delta_y) * image_value(image, x2, y2) + \
                      (1 - delta_x) * delta_y * image_value(image, x2, y2 + 1) + \
                      delta_x * (1 - delta_y) * image_value(image, x2 + 1, y2) + \
                      delta_x * delta_y * image_value(image, x2 + 1, y2 + 1)
            rot_image[int(j)][int(k)][0] = 1.3 * value
    return rot_image

def load_data(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        for k in range(0, 10000):
            elem = teX[k]
            # cv2.imwrite(os.path.join('images', '%05d-real.png' % (k,)), elem)
            elem1 = rotate(elem)
            teX[k] = elem1
            # cv2.imwrite(os.path.join('images', '%05d-modified.png' % (k,)), elem1)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def get_batch_data(batch_size, num_threads):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
