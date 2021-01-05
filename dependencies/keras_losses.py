# -*- coding: utf-8 -*-
"""
Module creating the custom losses/layer (since keras is not really permissive on the loss functions).
DISCLAIMER : We did not created those layers from scratch,
 we took it and sligthly adapted it from some repo that we quote (we may have sligly modified it) 
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers


class SphereFace(Layer):
    """
    Sphere face layer implementation which allows to compute the sphere loss.
    Source : https://github.com/4uiiurz1/keras-arcface
    """
    def __init__(self, n_classes=10, s=15.0, m=0.75, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(self.m * theta)
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class CenterLossLayer(Layer):
    """
    Center loss layer implementation which allows to compute the sphere loss.
    Source (slightly modified) :https://github.com/handongfeng/MNIST-center-loss/blob/master/centerLoss_MNIST.py
    """
    def __init__(self,n_classes=10,embeding_size = 512, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.embeding_size = embeding_size
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.n_classes, self.embeding_size),
                                       initializer='uniform',
                                       trainable=False)
       
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nxemb, x[1] is Nxclasses onehot, self.centers is classesxemb
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)


        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)
