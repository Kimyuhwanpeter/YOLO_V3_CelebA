# -*- coding:utf-8 -*-
from absl import logging
from tensorflow.python.keras.applications import resnet
from tensorflow.python.util.tf_export import keras_export

import tensorflow as tf
import numpy as np
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

ZeroPadding2D = tf.keras.layers.ZeroPadding2D
BatchNormalization = tf.keras.layers.BatchNormalization
LeakyReLU = tf.keras.layers.LeakyReLU
Conv2D = tf.keras.layers.Conv2D
Add = tf.keras.layers.Add
Lambda = tf.keras.layers.Lambda
UpSampling2D = tf.keras.layers.UpSampling2D
Concatenate = tf.keras.layers.Concatenate
l2 = tf.keras.regularizers.l2
Input = tf.keras.Input

@keras_export('keras.applications.resnet_v2.ResNet50V2',
              'keras.applications.ResNet50V2')
def ResNet50V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):
    """Instantiates the ResNet50V2 architecture."""

    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name='conv2')
        x = resnet.stack2(x, 128, 4, name='conv3')  # 36    conv3_block4
        x = resnet.stack2(x, 256, 6, name='conv4')  # 61
        return resnet.stack2(x, 512, 3, stride1=1, name='conv5')    # x

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        'resnet50v2',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation)

def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output



def YoloV3(size=None, channels=3, masks=yolo_anchor_masks, classes=80):

    x = inputs = Input([size, size, channels], name='input')
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    return tf.keras.Model(inputs, (x_36, x_61, x), name='yolov3')
    #return ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(size, size, channels),pooling=None)

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

