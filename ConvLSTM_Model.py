from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, Dense, Flatten, Masking, Softmax, MaxPooling2D, LeakyReLU


class ConvLSTMModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y, mask_value=-1):
        super().__init__()
        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        self.add(Masking(mask_value=mask_value))

        self.add(ConvLSTM2D(filters=4, kernel_size=(5, 5), activation='tanh', padding='same',
                            recurrent_activation='hard_sigmoid', return_sequences=True, name='convlstm2d_1'))
        self.add(MaxPooling3D(pool_size=(1, 3, 3), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', padding='same',
                            recurrent_activation='hard_sigmoid', return_sequences=True, name='convlstm2d_2'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), recurrent_activation='hard_sigmoid', padding='same'
                            , activation='tanh', return_sequences=True, name='convlstm2d_3'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), recurrent_activation='hard_sigmoid', padding='same'
                            , activation='tanh', return_sequences=True, name='convlstm2d_4'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), recurrent_activation='hard_sigmoid', padding='same'
                            , activation='tanh', return_sequences=False, name='convlstm2d_5'))
        self.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last'))

        self.add(Flatten(name='flatten_layer'))
        self.add(LeakyReLU())
        self.add(Dense(7, name='output'))
        self.add(Softmax())
