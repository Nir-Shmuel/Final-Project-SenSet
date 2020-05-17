from tensorflow.keras.optimizers import Nadam, Adam, SGD

from ConvLSTM_Model import ConvLSTMModel
from VideoDataGenerator import VideoDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
import os
import matplotlib.pyplot as plt
import glob

n_epochs = 150
batch_size = 4
root_path = '/tf/convlstm'
model_save_path = root_path + '/model'
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


def gen_and_save_graphs():
    if not os.path.exists(loss_save_path):
        os.mkdir(loss_save_path)
    # "Loss"
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
    plt.savefig(acc_save_path + r'\accuracy')
    plt.close()


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
                                     folder_name=data_root_folder, data_format='.%s' % videos_format)
val_generator = VideoDataGenerator(list_IDs=partition['validation'], dict_id_data=dict_id_data, batch_size=batch_size,
                                   data_format='.%s' % videos_format, folder_name=data_root_folder)
model = ConvLSTMModel(channels=3, pixels_x=96, pixels_y=96)
model.summary()

optimizer = SGD()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# train the model
callbacks_list = [
    callbacks.BaseLogger
]
history = model.fit(x=train_generator, validation_data=val_generator, epochs=n_epochs)
