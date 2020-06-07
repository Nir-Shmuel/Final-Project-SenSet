from tensorflow.keras import Sequential, Input, regularizers
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Softmax, BatchNormalization, \
    LeakyReLU, LSTM, TimeDistributed, Conv2D, Dropout, Conv3D, MaxPool3D, GlobalAveragePooling3D


def cnn_lstm(channels=3, pixels_x=96, pixels_y=96):
    model = Sequential()
    model.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), data_format='channels_last'),
                              name='conv1'))
    # self.add(TimeDistributed(BatchNormalization(), name='BN_1'))
    model.add(
        TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last'), name='MP_1'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_1'))

    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_2'))
    model.add(TimeDistributed(BatchNormalization(), name='BN_2'))
    model.add(
        TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='valid', data_format='channels_last'), name='MP_2'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_2'))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_3'))
    # self.add(TimeDistributed(BatchNormalization(), name='BN_3'))
    model.add(
        TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last'), name='MP_3'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_3'))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_4'))
    model.add(TimeDistributed(BatchNormalization(), name='BN_4'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_4'))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_5'))
    # self.add(TimeDistributed(BatchNormalization(), name='BN_5'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_5'))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Flatten(), name='flatten_layer'))
    model.add(TimeDistributed(BatchNormalization(), name='BN_6'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_6'))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(LSTM(units=512, return_sequences=False, dropout=0.5, activation='tanh'))
    # self.add(BatchNormalization())
    # self.add(LeakyReLU())
    # self.add(Dropout(0.5))
    model.add(Dense(units=7, name='output'))
    model.add(Softmax())
    return model


def cnn3d(channels=3, pixels_x=96, pixels_y=96):
    model = Sequential()
    model.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                     kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_1"))
    model.add(BatchNormalization(name="bn_1"))
    model.add(LeakyReLU(alpha=0.3, name="lr_1"))
    model.add(MaxPool3D(pool_size=(1, 2, 2), strides=None, padding='valid', name="mp_1"))
    #         self.add(layers.Dropout(0.2))

    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                     kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_2"))
    model.add(BatchNormalization(name="bn_2"))
    model.add(LeakyReLU(alpha=0.3, name="lr_2"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid', name="mp_2"))

    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                     kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_3"))
    model.add(BatchNormalization(name="bn_3"))
    model.add(LeakyReLU(alpha=0.3, name="lr_3"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid', name="mp_3"))
    #         self.add(layers.Dropout(0.15))

    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                     kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_4"))
    model.add(BatchNormalization(name="bn_4"))
    model.add(LeakyReLU(alpha=0.3, name="lr_4"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=None, padding='valid', name="mp_4"))

    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', use_bias=False,
                     kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), name="Conv_5"))

    model.add(Dropout(0.3))
    model.add(GlobalAveragePooling3D())
    model.add(Dense(7))
    model.add(Softmax())

    return model
