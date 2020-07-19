import glob

'''
Each video path is the shape of:
    ./data/<folder_name>/<partition>/<labels[i]>/<%04d><data_format>
Example:
    ./data/Cropped_Faces_CAER_npy/train/Anger/0001.npy

'''


def path2video_name(v_path, vf_len, id_len):
    return int(v_path[-(id_len + vf_len + 1): -(vf_len + 1)])


def map_data(data_path, emotions, videos_format):
    folders_structure = {
        'train': [emotions[i] for i in range(len(emotions))],
        'validation': [emotions[i] for i in range(len(emotions))],
        'test': [emotions[i] for i in range(len(emotions))]
    }
    partition = {
        'train': [],
        'validation': [],
        'test': []
    }
    dict_id_data = {}
    labels_name2val = {emotions[i]: i for i in range(len(emotions))}
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
    return partition, dict_id_data
