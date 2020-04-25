from ConvLSTM_Model import ConvLSTMModel

model = ConvLSTMModel(channels=3, pixels_x=96, pixels_y=96)
model.model.summary()
