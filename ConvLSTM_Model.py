from tensorflow.keras import Sequential, Input, regularizers
from tensorflow.keras.layers import MaxPooling3D, Dense, Flatten, Softmax, BatchNormalization, \
    LeakyReLU, LSTM, TimeDistributed, Conv2D, Dropout


class ConvLSTMModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y):
        super().__init__()

        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        self.add(TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), data_format='channels_last',
                                        kernel_regularizer=regularizers.l2(0.01)), name='conv1'))
        self.add(BatchNormalization())
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))
        self.add(LeakyReLU())

        self.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last',
                                        kernel_regularizer=regularizers.l2(0.01)), name='conv_2'))
        self.add(BatchNormalization())
        self.add(MaxPooling3D(pool_size=(1, 3, 3), padding='valid', data_format='channels_last'))
        self.add(LeakyReLU())

        self.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last',
                                        kernel_regularizer=regularizers.l2(0.01)), name='conv_3'))
        self.add(BatchNormalization())
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))
        self.add(LeakyReLU())
        self.add(Dropout(0.2))

        self.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last',
                                        kernel_regularizer=regularizers.l2(0.01)), name='conv_4'))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        self.add(Dropout(0.2))

        self.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last',
                                        kernel_regularizer=regularizers.l2(0.01)), name='conv_5'))
        self.add(BatchNormalization())
        self.add(Dropout(0.2))

        self.add(TimeDistributed(Flatten(), name='flatten_layer'))
        self.add(LSTM(units=512, return_sequences=False, dropout=0.5, activation='tanh',
                      kernel_regularizer=regularizers.l2(0.01)))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        self.add(Dropout(0.5))
        self.add(Dense(units=7, name='output'))
        self.add(Softmax())
