from tensorflow.keras import Sequential, Input, regularizers, Model, layers
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Softmax, BatchNormalization, \
    LeakyReLU, LSTM, TimeDistributed, Conv2D, Dropout, Conv3D, MaxPool3D, GlobalAveragePooling3D


def cnn_lstm(channels=3, pixels_x=96, pixels_y=96, output_size=7):
    model = Sequential(name='cnn_lstm')
    model.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input_cnnlstm'))
    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), data_format='channels_last'),
                              name='conv1'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last'),
                              name='MP_1'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_1'))

    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_2'))
    model.add(TimeDistributed(BatchNormalization(), name='BN_2'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='valid', data_format='channels_last'),
                              name='MP_2'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_2'))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_3'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_last'),
                              name='MP_3'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_3'))

    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_4'))
    model.add(TimeDistributed(BatchNormalization(), name='BN_4'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_4'))
    model.add(TimeDistributed(Dropout(0.3), name='Dropout_4'))

    model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'),
                              name='conv_5'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_5'))
    model.add(TimeDistributed(Dropout(0.3, name='Dropout_5')))

    model.add(TimeDistributed(Flatten(), name='flatten_layer'))
    model.add(TimeDistributed(BatchNormalization(), name='BN_6'))
    model.add(TimeDistributed(LeakyReLU(), name='LR_6'))
    model.add(LSTM(units=512, return_sequences=False, dropout=0.5, activation='tanh'))
    model.add(LeakyReLU(name='LR_7'))
    model.add(Dense(units=128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization(name='BN_8'))
    model.add(LeakyReLU(name='LR_8'))
    model.add((Dropout(0.5, name='Dropout_8')))
    model.add(Dense(units=output_size, name='output'))
    model.add(Softmax())

    return model


def cnn3d(channels=3, pixels_x=96, pixels_y=96, output_size=7):
    model = Sequential(name='conv3d')
    model.add(Input(shape=(None, pixels_x, pixels_y, channels), name='input_conv3d'))
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                     kernel_initializer='glorot_normal', name="Conv_1"))
    model.add(TimeDistributed(BatchNormalization(), name="bn_1"))
    model.add(TimeDistributed(LeakyReLU(alpha=0.3), name="lr_1"))

    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                     kernel_initializer='glorot_normal', name="Conv_2"))
    model.add(TimeDistributed(BatchNormalization(), name="bn_2"))
    model.add(TimeDistributed(LeakyReLU(alpha=0.3), name="lr_2"))

    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                     kernel_initializer='glorot_normal', name="Conv_3"))
    model.add(TimeDistributed(BatchNormalization(), name="bn_3"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='valid', name="mp_3"))
    model.add(TimeDistributed(LeakyReLU(alpha=0.3), name="lr_3"))

    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 2, 2), padding='same',
                     kernel_initializer='glorot_normal', name="Conv_4"))
    model.add(TimeDistributed(BatchNormalization(), name="bn_4"))
    model.add(TimeDistributed(LeakyReLU(alpha=0.3), name="lr_4"))
    model.add(TimeDistributed(Dropout(0.25), name="dropout_4"))

    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                     kernel_initializer='glorot_normal', name="Conv_5"))
    model.add(TimeDistributed(BatchNormalization(), name="bn_5"))
    model.add(TimeDistributed(LeakyReLU(alpha=0.3), name="lr_5"))
    model.add(TimeDistributed(Dropout(0.3), name="dropout_5"))

    model.add(GlobalAveragePooling3D())
    model.add(BatchNormalization(name="bn_6"))
    model.add(LeakyReLU(alpha=0.3, name="lr_6"))
    model.add(Dropout(0.5, name="dropout_6"))
    model.add(Dense(units=128, kernel_regularizer=regularizers.l2(0.005)))
    model.add(BatchNormalization(name="bn_7"))
    model.add(LeakyReLU(alpha=0.3, name="lr_7"))
    model.add(Dropout(0.5, name="dropout_7"))
    model.add(Dense(output_size, name="output"))
    model.add(Softmax())

    return model


def ensemble(models=None, output_size=7):
    for i, model in enumerate(models, 1):
        model.trainable = False
        for j, layer in enumerate(model.layers, 1):
            layer._name = model.name + str(i) + '_' + layer.name

    inputs = [model.input for model in models]
    for i, mdl in enumerate(models, 1):
        mdl.layers[0]._name = str(i)
    ensemble_outputs = layers.concatenate([model.output for model in models])
    output = Dense(output_size, activation='softmax')(ensemble_outputs)
    combined = Model(inputs=inputs, outputs=output)
    return combined
