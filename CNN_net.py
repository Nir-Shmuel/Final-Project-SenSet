import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential, Input


class CnnModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y):
        super().__init__()
        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        self.add(
            layers.Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_1"))
        self.add(layers.BatchNormalization(name="bn_1"))
        self.add(layers.LeakyReLU(alpha=0.3, name="lr_1"))
        self.add(layers.MaxPool3D(pool_size=(1, 2, 2), strides=None, padding='valid', name="mp_1"))
#         self.add(layers.Dropout(0.2))

        self.add(
            layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_2"))
        self.add(layers.BatchNormalization(name="bn_2"))
        self.add(layers.LeakyReLU(alpha=0.3, name="lr_2"))
        self.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid', name="mp_2"))

        self.add(
            layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_3"))
        self.add(layers.BatchNormalization(name="bn_3"))
        self.add(layers.LeakyReLU(alpha=0.3, name="lr_3"))
        self.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid', name="mp_3"))
#         self.add(layers.Dropout(0.15))

        self.add(
            layers.Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_4"))
        self.add(layers.BatchNormalization(name="bn_4"))
        self.add(layers.LeakyReLU(alpha=0.3, name="lr_4"))
        self.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid', name="mp_4"))

        self.add(
            layers.Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_5"))

        self.add(layers.Dropout(0.3))
        self.add(layers.GlobalAveragePooling3D())
        self.add(layers.Dense(7))
        self.add(layers.Softmax())
