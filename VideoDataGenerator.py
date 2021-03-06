import numpy as np
from tensorflow import keras
import random

'''
Each video path is the shape of:
    ./data/<folder_name>/<partition>/<labels[i]>/<%04d><data_format>
Example:
    ./data/Cropped_Faces_CAER_npy/train/Anger/0001.npy

'''

'''
listIDs: list that contains the IDs of videos in the given partition. shape: <partition>-<emotion>-<video_number>.
dict_id_data:  dictionary that contains the metadata for each video in the given partition.
                keys: values of listIDs.
                values: a dictionary with keys: 1) 'emotion' - the video's emotion.
                                                2) 'video_name' - the name of of the video file.
                                                3) 'label_val' - the y_true value of the video.

batch_size: number of videos in each batch.
dim: the height and width of a frame.
n_channels: number of channels in a frame.
n_inputs: number of inputs of the network.
padding_val: value of the added frames when the video has less than the minimum timesteps.
timesteps:  when not None, each video is set to have timesteps number of frames.
            if video has less than timesteps => it is padded with padding_val.
            if video has more than timesteps => randomly pick a timesteps long part of the video.
n_classes: number of classes to be classified.
shuffle: if true, the order of videos will be shuffled on each epoch end.
folder_name: the path to root folder of the data.
partition: the partition to take the data from: 'train' | 'validation' | 'test'. 
data_format: the format of the data.
flip_horizontal: if true, the data will be flipped horizontally in probability of flip_prob. 
flip_vertical: if true, the data will be flipped vertically in probability of flip_prob.
flip_prob: the probability to flip the data.
'''


class VideoDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, dict_id_data, folder_name, batch_size=1, dim=(96, 96), n_channels=3, n_inputs=1,
                 padding_val=0, timesteps=None,
                 n_classes=7, shuffle=True, partition=None,
                 data_format='.npy', flip_horizontal=False, flip_vertical=False, flip_prob=0.5):
        self.list_IDs = list_IDs
        self.dict_id_data = dict_id_data
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_inputs = n_inputs
        self.padding_val = padding_val
        self.timesteps = timesteps
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_format = data_format
        self.folder_name = folder_name
        if partition not in {'train', 'validation', 'test'}:
            raise ValueError('partition should be: train, validation or test')
        self.partition = partition
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        if 0 <= flip_prob <= 1:
            self.flip_prob = flip_prob
        self.on_epoch_end()

    # Returns the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[i] for i in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.flip_vertical and np.random.uniform(0, 1) < self.flip_prob:
            X = np.flip(X, axis=2)

        if self.flip_horizontal and np.random.uniform(0, 1) < self.flip_prob:
            X = np.flip(X, axis=3)

        return [X] * self.n_inputs, y, [None]

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size samples
    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, max_n_frames, *dim, n_channels)
        # Initialization

        videos_list = []
        # Generate data
        y = np.empty(shape=(self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            video_emotion = self.dict_id_data[ID]['emotion']
            video_name = self.dict_id_data[ID]['video_name']
            file_path = '%s/%s/%s/%04d%s' % (
                self.folder_name, self.partition, video_emotion, video_name, self.data_format)
            video = np.load(file=file_path)
            y[i] = self.dict_id_data[ID]['label_val']
            videos_list.append(video)

        if self.timesteps is None:
            max_n_frames = np.max([v.shape[0] for v in videos_list])
            max_n_frames = min(max_n_frames, 20)
        else:
            max_n_frames = self.timesteps
        r_trunc = self.random_truncating(videos_list, max_n_frames)
        # pad videos with self.padding_val
        X = keras.preprocessing.sequence.pad_sequences(sequences=r_trunc, padding='post', maxlen=max_n_frames,
                                                       value=self.padding_val)

        y_onehot = keras.utils.to_categorical(y, num_classes=self.n_classes)
        # return labels in one_hot form
        return X, y_onehot

    def random_truncating(self, video_list, n_frames):
        padded = []
        for video in video_list:
            if video.shape[0] > n_frames:
                i = random.randint(0, video.shape[0] - n_frames)
                padded.append(video[i:i + n_frames])
            else:
                padded.append(video)
        return padded
