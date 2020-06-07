from tensorflow.keras.optimizers import SGD
import tensorflow.keras as keras
import Models
from VideoDataGenerator import VideoDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import sklearn

n_epochs = 150
batch_size = 16
root_path = '/tf'
model_save_path = root_path + '/model'
model_save_name = 'cnnlstm.hdf5'
loss_save_path = root_path + '/loss'
acc_save_path = root_path + '/accuracy'
data_root_folder = '/tf/data/Cropped_Faces_CAER_npy'
videos_format = 'npy'

emotions = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad')
folders_structure = {
    'train': [emotions[i] for i in range(len(emotions))],
    'validation': [emotions[i] for i in range(len(emotions))],
    'test': [emotions[i] for i in range(len(emotions))]
}
labels_name2val = {emotions[i]: i for i in range(len(emotions))}
partition = {
    'train': [],
    'validation': [],
    'test': []
}

dict_id_data = {}


def path2video_name(v_path, vf_len, id_len):
    return int(v_path[-(id_len + vf_len + 1): -(vf_len + 1)])


def map_data(data_path):
    for par in folders_structure:
        par_path = '%s/%s' % (data_path, par)
        for emotion in folders_structure[par]:
            emotion_path = '%s/%s' % (par_path, emotion)
            video_paths = glob.glob('%s/*%s' % (emotion_path, videos_format))
            for video_path in video_paths:
                video_name = path2video_name(v_path=video_path, vf_len=len(videos_format), id_len=4)
                video_id = '%s-%s-%04d' % (par, emotion, video_name)
                partition[par].append(video_id)
                dict_id_data[video_id] = {'emotion': emotion,
                                          'video_name': video_name,
                                          'label_val': labels_name2val[emotion],
                                          }


map_data(data_root_folder)

train_generator = VideoDataGenerator(list_IDs=partition['train'], dict_id_data=dict_id_data, batch_size=batch_size,
                                     folder_name=data_root_folder, partition='train', flip_vertical=True,
                                     flip_horizontal=True, flip_prob=0.5)

val_generator = VideoDataGenerator(list_IDs=partition['validation'], dict_id_data=dict_id_data, batch_size=batch_size,
                                   folder_name=data_root_folder, partition='validation', flip_vertical=True,
                                   flip_horizontal=True, flip_prob=0.5)

test_generator = VideoDataGenerator(list_IDs=partition['test'], dict_id_data=dict_id_data, batch_size=batch_size,
                                    folder_name=data_root_folder, partition='test')

if os.path.exists(os.path.join(model_save_path, model_save_name)):
    print('Loading model')
    model = load_model(filepath=os.path.join(model_save_path, model_save_name))
else:
    print("Creating CNN+LSTM model.")
    model = Models.cnn3d(channels=3, pixels_x=96, pixels_y=96)

model.summary()
optimizer = SGD(learning_rate=0.1, clipnorm=1)
loss = keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=15),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, min_delta=5e-3, min_lr=1e-5),
    callbacks.ModelCheckpoint(filepath=os.path.join(model_save_path, model_save_name), verbose=0, save_best_only=True)
]

# calculate class weights
y = [dict_id_data[i]['label_val'] for i in dict_id_data.keys()]
class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

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
plt.savefig(loss_save_path + '/loss_cnnlstm')
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
plt.savefig(acc_save_path + '/accuracy_cnnlstm')
plt.close()

print('predicting test set')

# print confusion matrix
y_true = np.concatenate([np.argmax(test_generator[i][1], axis=1) for i in range(len(test_generator))])
y_pred = np.argmax(model.predict(x=test_generator, steps=len(test_generator)), axis=1)

print('creating and saving confusion matrix')

cm = pd.DataFrame(data=confusion_matrix(y_true=y_true, y_pred=y_pred),
                  index=['True: %s' % emotion for emotion in emotions],
                  columns=['Pred: %s' % emotion for emotion in emotions])
cm.to_csv(path_or_buf=root_path + '/confusion_matrix_cnnlstm.csv')
print(cm)
print(classification_report(y_true=y_true, y_pred=y_pred, target_names=[emotion for emotion in emotions]))
