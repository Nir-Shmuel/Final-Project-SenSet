from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D, Dense, Flatten, \
    Masking


class ConvLSTMModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y, mask_value=-1):
        super().__init__()
        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        assert self.output_shape == (None, None, pixels_x, pixels_y, channels)
        self.add(Masking(mask_value=mask_value))
        assert self.output_shape == (None, None, pixels_x, pixels_y, channels)
        self.add(ConvLSTM2D(filters=20, kernel_size=(3, 3), data_format='channels_last', activation='tanh',
                            padding='same', recurrent_activation='hard_sigmoid', return_sequences=True,
                            name='first_convlstm2d'))
        assert self.output_shape == (None, None, pixels_x, pixels_y, 20)
        self.add(BatchNormalization(name='first_bn'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last', name='first_mp'))
        assert self.output_shape == (None, None, pixels_x // 2, pixels_y // 2, 20)
        self.add(
            ConvLSTM2D(filters=10, kernel_size=(3, 3), data_format='channels_last', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=True, name='second_convlstm2d'))
        assert self.output_shape == (None, None, pixels_x // 2, pixels_y // 2, 10)
        self.add(BatchNormalization(name='second_bn'))
        self.add(MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_last', name='second_mp'))
        assert self.output_shape == (None, None, pixels_x // 6, pixels_y // 6, 10)
        self.add(
            ConvLSTM2D(filters=5, kernel_size=(3, 3), data_format='channels_last', recurrent_activation='hard_sigmoid'
                       , activation='tanh', padding='same', return_sequences=False, name='third_convlstm2d'))
        assert self.output_shape == (None, pixels_x // 6, pixels_y // 6, 5)
        self.add(Flatten(name='flatten_layer'))
        self.add(Dense(512, activation='relu', name='first_fc'))
        self.add(Dense(32, activation='relu', name='second_fc'))
        self.add(Dense(7, activation='softmax', name='third_fc'))
