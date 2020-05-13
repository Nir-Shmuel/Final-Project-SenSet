from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, Dense, Flatten, Masking, Softmax, BatchNormalization, \
    LeakyReLU, LSTM, TimeDistributed


class ConvLSTMModel(Sequential):
    def __init__(self, channels, pixels_x, pixels_y, mask_value=-1):
        super().__init__()
        self.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input'))
        self.add(Masking(mask_value=mask_value))

        self.add(ConvLSTM2D(filters=4, kernel_size=(5, 5), activation='tanh', padding='same',
                            recurrent_activation='hard_sigmoid', return_sequences=True, name='convlstm2d_1'))
        self.add(MaxPooling3D(pool_size=(1, 3, 3), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=8, kernel_size=(5, 5), activation='tanh', padding='same',
                            recurrent_activation='hard_sigmoid', return_sequences=True, name='convlstm2d_2'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), recurrent_activation='hard_sigmoid', padding='same'
                            , activation='tanh', return_sequences=True, name='convlstm2d_3'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), recurrent_activation='hard_sigmoid', padding='same'
                            , activation='tanh', return_sequences=True, name='convlstm2d_4'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), recurrent_activation='hard_sigmoid', padding='same'
                            , activation='tanh', return_sequences=True, name='convlstm2d_5'))
        self.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_last'))

        self.add(ConvLSTM2D(filters=128, kernel_size=(1, 1), recurrent_activation='hard_sigmoid', padding='same'
                            , activation='tanh', return_sequences=True, name='convlstm2d_6'))

        self.add(TimeDistributed(Flatten(), name='flatten_layer'))
        self.add(LSTM(units=32, return_sequences=False))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        self.add(Dense(7, name='output'))
        self.add(Softmax())