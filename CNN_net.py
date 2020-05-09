import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential, Input


class CnnModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y):
        super().__init__()
        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        self.add(
            layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
        self.add(layers.LeakyReLU(alpha=0.3))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool3D(pool_size=(1, 2, 2), strides=None, padding='valid'))

        self.add(
            layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
        self.add(layers.LeakyReLU(alpha=0.3))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

        self.add(
            layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
        self.add(layers.LeakyReLU(alpha=0.3))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

        self.add(
            layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
        self.add(layers.LeakyReLU(alpha=0.3))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

        self.add(
            layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2))
        self.add(layers.LeakyReLU(alpha=0.3))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid'))

        self.add(layers.Flatten())
        self.add(layers.Dense(256))
        self.add(layers.LeakyReLU(alpha=0.3))

        self.add(layers.Dense(7))
        self.add(layers.Softmax(-1))

        return self
