import numpy as np
from tensorflow import keras

'''
Each video path is the shape of:
    ./data/<folder_name>/<partition>/<labels[i]>/<%04d><data_format>
Example:
    ./data/Cropped_Faces_CAER_npy/train/Anger/0001.npy
    
'''


class VideoDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, dict_id_data, batch_size=32, dim=(96, 96), n_channels=3, padding_val=-1,
                 n_classes=7, shuffle=True, folder_name='./data/Cropped_Faces_CAER_npy', partition='train',
                 data_format='npy'):
        self.list_IDs = list_IDs
        self.dict_id_data = dict_id_data
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.padding_val = padding_val
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_format = data_format
        self.folder_name = folder_name
        self.partition = partition
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

        return X, y, [None]

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size samples
    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        # Initialization
        max_n_frames = 0

        # list of tuple (ID, video)
        videos_list = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            video_emotion = self.dict_id_data[ID]['emotion']
            video_name = self.dict_id_data[ID]['video_name']
            file_path = '%s/%s/%s/%04d%s' % (
                self.folder_name, self.partition, video_emotion, video_name, self.data_format)
            video = np.load(file=file_path)
            max_n_frames = max(max_n_frames, video.shape[0])
            videos_list.append((ID, video))

        X = np.empty(shape=(self.batch_size, max_n_frames, *self.dim, self.n_channels))
        y = np.empty(shape=(self.batch_size), dtype=int)
        for i, tuple in enumerate(videos_list):
            ID, video = tuple
            X[i] = self.pad_video(video, max_n_frames)
            y[i] = self.dict_id_data[ID]['label_val']

        # return labels in one_hot form
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    # pad the video with frames filled with padding_val
    def pad_video(self, video, max_frames):
        padded_video = np.full(shape=(max_frames, *(video.shape[1:])), fill_value=self.padding_val)
        padded_video[:video.shape[0]] = video
        return padded_video
