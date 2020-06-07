from tensorflow.keras import Sequential, Input, regularizers
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Softmax, BatchNormalization, \
    LeakyReLU, LSTM, TimeDistributed, Conv2D, Dropout


class ConvLSTMModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y):
        super().__init__()

        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))

        self.add(TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), data_format='channels_last'),
                                 name='conv1'))
        # self.add(TimeDistributed(BatchNormalization(), name='BN_1'))
        self.add(
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last'), name='MP_1'))
        self.add(TimeDistributed(LeakyReLU(), name='LR_1'))

        self.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_2'))
        self.add(TimeDistributed(BatchNormalization(), name='BN_2'))
        self.add(
            TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='valid', data_format='channels_last'), name='MP_2'))
        self.add(TimeDistributed(LeakyReLU(), name='LR_2'))

        self.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_3'))
        # self.add(TimeDistributed(BatchNormalization(), name='BN_3'))
        self.add(
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last'), name='MP_3'))
        self.add(TimeDistributed(LeakyReLU(), name='LR_3'))
        self.add(TimeDistributed(Dropout(0.3)))

        self.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_4'))
        self.add(TimeDistributed(BatchNormalization(), name='BN_4'))
        self.add(TimeDistributed(LeakyReLU(), name='LR_4'))
        self.add(TimeDistributed(Dropout(0.3)))

        self.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_5'))
        # self.add(TimeDistributed(BatchNormalization(), name='BN_5'))
        self.add(TimeDistributed(LeakyReLU(), name='LR_5'))
        self.add(TimeDistributed(Dropout(0.3)))

        self.add(TimeDistributed(Flatten(), name='flatten_layer'))
        self.add(TimeDistributed(BatchNormalization(), name='BN_6'))
        self.add(TimeDistributed(LeakyReLU(), name='LR_6'))
        self.add(TimeDistributed(Dropout(0.5)))
        self.add(LSTM(units=512, return_sequences=False, dropout=0.5, activation='tanh'))
        # self.add(BatchNormalization())
        # self.add(LeakyReLU())
        # self.add(Dropout(0.5))
        self.add(Dense(units=7, name='output'))
        self.add(Softmax())
