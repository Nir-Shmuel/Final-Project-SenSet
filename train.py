from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras as keras
from ConvLSTM_Model import ConvLSTMModel
from VideoDataGenerator import VideoDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
import os
import matplotlib.pyplot as plt
import glob

n_epochs = 100
batch_size = 16
root_path = '/tf/convlstm'
model_save_path = root_path + '/model'
loss_save_path = root_path + '/loss'
acc_save_path = root_path + '/accuracy'
data_root_folder = '/tf/data/Cropped_Faces_CAER_npy'
saved_model = None
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

# set generators with default values
train_generator = VideoDataGenerator(list_IDs=partition['train'], dict_id_data=dict_id_data, batch_size=batch_size,
                                     folder_name=data_root_folder, partition='train')
val_generator = VideoDataGenerator(list_IDs=partition['validation'], dict_id_data=dict_id_data, batch_size=batch_size,
                                   folder_name=data_root_folder, partition='validation')
if saved_model is not None:
    print("Loading model %s" % saved_model)
    model = load_model(saved_model)
else:
    print("Creating LSTM model.")
    model = ConvLSTMModel(channels=3, pixels_x=96, pixels_y=96)
model.summary()

optimizer = RMSprop()
loss = keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# train the model
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', min_delta=5e-3, patience=10),
    callbacks.ModelCheckpoint(filepath=os.path.join(model_save_path, 'model.{epoch:03d}-{val_loss:.3f}.hdf5'),
                              verbose=1,
                              save_best_only=True)
]
history = model.fit(x=train_generator, validation_data=val_generator, epochs=n_epochs, callbacks=callbacks_list)

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
plt.savefig(loss_save_path + '/losses')
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
plt.savefig(acc_save_path + '/accuracy')
plt.close()
