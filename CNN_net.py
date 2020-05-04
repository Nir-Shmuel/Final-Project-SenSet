import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def make_CNN_model(input_shape):
    model = tf.keras.Sequential()
    model.add(
        layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False, input_shape=input_shape,
                      kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D(pool_size=(1, 2, 2), strides=None, padding='valid'))

    model.add(
        layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                      kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(
        layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                      kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(
        layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                      kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(
        layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                      kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.3))

    model.add(layers.Dense(7))
    model.add(layers.Softmax(-1))

    return model
