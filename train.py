from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from dotenv import load_dotenv
import tensorflow.keras as keras

from DataMapper import map_data
import Models
from VideoDataGenerator import VideoDataGenerator
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

load_dotenv(dotenv_path='dotenv')

n_epochs = 200
batch_size = 64
root_path = os.getenv('ROOT_PATH')
model_save_path = os.path.join(root_path, os.getenv('MODEL_SAVE_DIR'))
model_save_name = os.getenv('MODEL_SAVE_NAME')
loss_save_path = os.path.join(root_path, os.getenv('LOSS_SAVE_DIR'))
acc_save_path = os.path.join(root_path, os.getenv('ACC_SAVE_DIR'))
cm_save_path = os.path.join(root_path, os.getenv('CONFUSION_MATRIX_SAVE_DIR'))
data_root_folder = os.getenv('DATA_ROOT_FOLDER')
videos_format = os.getenv('VIDEOS_FORMAT')

emotions = ('Happy', 'Neutral', 'Sad')
name_optimizer = ('SGD', SGD(learning_rate=0.01, clipnorm=1))

partition, dict_id_data = map_data(data_path=data_root_folder, emotions=emotions, videos_format=videos_format)

train_generator = VideoDataGenerator(list_IDs=partition['train'], dict_id_data=dict_id_data,
                                     folder_name=data_root_folder, n_classes=len(emotions), batch_size=batch_size,
                                     partition='train', flip_vertical=True, flip_horizontal=True, flip_prob=0.5)

val_generator = VideoDataGenerator(list_IDs=partition['validation'], dict_id_data=dict_id_data,
                                   folder_name=data_root_folder, n_classes=len(emotions), batch_size=batch_size,
                                   partition='validation', flip_vertical=True, flip_horizontal=True, flip_prob=0.5)

test_generator = VideoDataGenerator(list_IDs=partition['test'], dict_id_data=dict_id_data,
                                    folder_name=data_root_folder,
                                    n_classes=len(emotions), shuffle=False, batch_size=batch_size, partition='test')

print("Creating CNN+LSTM model.")
model = Models.cnn_lstm(channels=3, pixels_x=96, pixels_y=96, output_size=len(emotions))

# print("Creating CNN3D model.")
# model = Models.cnn3d(channels=3, pixels_x=96, pixels_y=96, output_size=len(emotions))

model.summary()

name, optimizer = name_optimizer

file_name_end = '_%s_%s' % (len(emotions), name)

loss = keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=15),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=5e-3,
                                min_lr=1e-5),
    callbacks.ModelCheckpoint(filepath=os.path.join(model_save_path, model_save_name) + file_name_end + '.hdf5',
                              verbose=0, save_best_only=True)
]
# cnnlstm
class_weights = {0: 1, 1: .6, 2: 3.8}
# conv3d
# class_weights={0: 1, 1: .6, 2: 4.}

# train the model
history = model.fit(x=train_generator, validation_data=val_generator, epochs=n_epochs, callbacks=callbacks_list,
                    class_weight=class_weights)

'''Generate Loss & Accuracy graphs'''
# "Loss"
if not os.path.exists(loss_save_path):
    os.mkdir(loss_save_path)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(os.path.join(loss_save_path, os.getenv('LOSS_SAVE_NAME')) + file_name_end)
plt.close()
#  "Accuracy"
if not os.path.exists(acc_save_path):
    os.mkdir(acc_save_path)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(os.path.join(acc_save_path, os.getenv('ACC_SAVE_NAME')) + file_name_end)
plt.close()

print('predicting test set')

# print confusion matrix
y_true = np.concatenate([np.argmax(test_generator[i][1], axis=1) for i in range(len(test_generator))])
y_pred = np.argmax(model.predict(x=test_generator, steps=len(test_generator)), axis=1)

print('creating and saving confusion matrix')
cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[i for i in range(len(emotions))])
print(cm.shape)
index = ['True: %s' % emotion for emotion in emotions]
columns = ['Pred: %s' % emotion for emotion in emotions]
df = pd.DataFrame(data=cm,
                  index=index,
                  columns=columns)
df.to_csv(path_or_buf=os.path.join(cm_save_path, os.getenv('CONFUSION_MATRIX_NAME')) + file_name_end + '.csv')
print(df)
print(classification_report(y_true=y_true, y_pred=y_pred, target_names=[emotion for emotion in emotions]))
