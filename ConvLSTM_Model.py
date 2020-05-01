from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D, TimeDistributed, Dense, Flatten, \
    Masking


class ConvLSTMModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y,mask_value=-1):
        super().__init__()
        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        self.add(Masking(mask_value=mask_value))
        self.add(
            ConvLSTM2D(filters=20, kernel_size=(3, 3), data_format='channels_last', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=True, name='first_convlstm2d'))
        self.add(BatchNormalization(name='first_bn'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last', name='first_mp'))
        self.add(
            ConvLSTM2D(filters=10, kernel_size=(3, 3), data_format='channels_last', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=True, name='second_convlstm2d'))
        self.add(BatchNormalization(name='second_bn'))
        self.add(
            MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_last', name='second_mp'))
        self.add(
            ConvLSTM2D(filters=5, kernel_size=(3, 3), data_format='channels_last', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=True, name='third_convlstm2d'))
        self.add(
            MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last', name='third_mp'))
        self.add(TimeDistributed(Flatten(name='flatten_layer')))
        self.add(TimeDistributed(Dense(512, ), name='first_fc'))
        self.add(TimeDistributed(Dense(32, ), name='second_fc'))
        self.add(TimeDistributed(Dense(7, activation='softmax'), name='third_fc'))
