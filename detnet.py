import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def res_block(kernel, filters, input, pool=True):
    x = input
    if pool:
        x = layers.MaxPool2D()(x)
    x1 = layers.Conv2D(filters=filters, kernel_size=kernel, padding="same")(x)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv2D(filters=filters, kernel_size=kernel, padding="same")(x1)
    x2 = layers.BatchNormalization()(x2)
    x3 = layers.Conv2D(filters=filters, kernel_size=kernel, padding="same")(x2)
    x3 = layers.BatchNormalization()(x3)
    x = layers.Add()([x1, x3])
    return x


def res_block_up(kernel, filters, input):
    x = input
    x = layers.UpSampling2D()(x)
    x1 = layers.Conv2D(filters=filters, kernel_size=kernel, padding="same")(x)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv2D(filters=filters, kernel_size=kernel, padding="same")(x1)
    x2 = layers.BatchNormalization()(x2)
    x3 = layers.Conv2D(filters=filters, kernel_size=kernel, padding="same")(x2)
    x3 = layers.BatchNormalization()(x3)
    x = layers.Add()([x1, x3])
    return x


def detnet(input_shape, kernel=3, filters=16):
    inputs = layers.Input(input_shape)
    block1 = res_block(kernel, filters, inputs, pool=False)
    block2 = res_block(kernel, filters * 2, block1)
    block3 = res_block(kernel, filters * 4, block2)
    block4 = res_block_up(kernel, filters * 2, block3)
    block5 = res_block_up(kernel, filters, block4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(block5)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model


def dice_coef2(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef2(y_true, y_pred)
