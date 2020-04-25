from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D, TimeDistributed, Dense, Flatten


class ConvLSTMModel:
    def __init__(self, channels, pixels_x, pixels_y):
        self.model = Sequential()
        self.model.add(Input(shape=(None, channels, pixels_x, pixels_y), name='input'))
        self.model.add(
            ConvLSTM2D(filters=20, kernel_size=(3, 3), data_format='channels_first', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=True, name='first_convlstm2d'))
        self.model.add(BatchNormalization(name='first_bn'))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first', name='first_mp'))
        self.model.add(
            ConvLSTM2D(filters=10, kernel_size=(3, 3), data_format='channels_first', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=True, name='second_convlstm2d'))
        self.model.add(BatchNormalization(name='second_bn'))
        self.model.add(
            MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_first', name='second_mp'))
        self.model.add(
            ConvLSTM2D(filters=5, kernel_size=(3, 3), data_format='channels_first', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=True, name='third_convlstm2d'))
        self.model.add(
            MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first', name='third_mp'))
        self.model.add(TimeDistributed(Flatten(name='flatten_layer')))
        self.model.add(TimeDistributed(Dense(512, ), name='first_fc'))
        self.model.add(TimeDistributed(Dense(32, ), name='second_fc'))
        self.model.add(TimeDistributed(Dense(7, activation='softmax'), name='third_fc'))
