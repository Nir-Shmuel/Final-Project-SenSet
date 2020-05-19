from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import MaxPooling3D, Dense, Flatten, Softmax, BatchNormalization, \
    LeakyReLU, LSTM, TimeDistributed, Conv2D, Dropout


class ConvLSTMModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y):
        super().__init__()

        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        self.add(TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), data_format='channels_last'),
                                 name='conv1'))
        self.add(BatchNormalization())
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))
        self.add(LeakyReLU())

        self.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_2'))
        self.add(BatchNormalization())
        self.add(MaxPooling3D(pool_size=(1, 3, 3), padding='valid', data_format='channels_last'))
        self.add(LeakyReLU())

        self.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_3'))
        self.add(BatchNormalization())
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))
        self.add(LeakyReLU())

        self.add(TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_4'))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        self.add(Dropout(0.2))

        self.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                                 name='conv_5'))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        self.add(Dropout(0.2))

        self.add(TimeDistributed(Flatten(), name='flatten_layer'))
        self.add(LSTM(units=128, return_sequences=True))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        self.add(Dropout(0.5))
        self.add(LSTM(units=64, return_sequences=False))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        self.add(Dropout(0.5))
        self.add(Dense(7, name='output'))
        self.add(Softmax())
