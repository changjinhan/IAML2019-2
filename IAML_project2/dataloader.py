import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import os
from signal_process import tf_Signal_Process
from tqdm import tqdm

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_record(example_proto, length, label_names):
    # Parse record from TFRecord file
    feature_description = {
        'track': tf.io.FixedLenFeature([length], tf.float32),
    }
    for label in label_names:
        feature_description[label] = tf.io.FixedLenFeature([], tf.int64)

    parsed = tf.io.parse_single_example(example_proto, feature_description)
    track = parsed['track']

    labels = list()
    for i in range(len(label_names)):
        labels.append(parsed[label_names[i]])
    labels = tf.cast(labels, dtype=tf.int64)

    return track, labels

class DataLoader(object):
    def __init__(self, metadata_path='metadata.csv', audio_dir='music_dataset', data_dir='data', split='training', signal_process='tf_stft', sr=22050, duration=30.0,):
        '''
        :param file_path: file path for metadata.csv
        :param audio_dir: audio data directory
        :param split: 'training' / 'validation' / 'test' mode
        :param signal_process: 'your_own_way', 'tf_stft', 'tf_mel_spectrogram', 'tf_log_mel_spectrogram', 'tf_mfcc',
                               'stft', 'cqt', 'chroma_cqt', 'chroma_cens', 'chroma_stft', 'rms', 'mel_spectrogram', 'mfcc'
        '''
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.track_column_name = 'PATH'
        self.label_names = ['happy', 'film', 'energetic', 'relaxing', 'emotional', 'melodic', 'dark', 'epic', 'dream', 'love']
        self.split = split
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.sr = sr
        self.duration = duration
        self.length = int(sr * duration)
        self.signal_process = signal_process

        # csv meta data load
        self.metadata_df = pd.read_csv(self.metadata_path)
        # self.label_dict = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.label_column_name].values)))}
        if self.split == 'training':
            self.metadata_df = self.metadata_df[self.metadata_df['split'] == 'training']
        elif self.split == 'validation':
            self.metadata_df = self.metadata_df[self.metadata_df['split'] == 'validation']
        elif self.split == 'test':
            self.metadata_df = self.metadata_df[self.metadata_df['split'] == 'test']
        print('===== Successfully %s meta data loaded' % self.split)

    def dataset(self):
        """
        :return: dataset with
                 signal processed audio feature [feature_size, sequence_length], integer label (0~7)
        """
        # save loaded audio, label or load data
        save_dir = os.path.join(self.data_dir, self.split)

        track_ids = self.metadata_df[self.track_column_name].to_list()
        save_paths = [os.path.join(save_dir, '%07d.tfrecord' % int(i.split('/')[1].split('.')[0])) for i in track_ids]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print("===== Preparing %s dataset! This could take a while..." % self.split)

            labels_dict = dict()
            for label in self.label_names:
                labels_dict[label] = self.metadata_df[label].to_list()

            for i in tqdm(range(len(track_ids))):

                # tfrecord writer
                save_path = save_paths[i]
                writer = tf.io.TFRecordWriter(save_path)

                # mp3 audio file load
                track = self.audio_load(filepath=self.get_audio_path(self.audio_dir, track_ids[i]), sr=self.sr, duration= self.duration)

                # track should be mono, and length have to be same with self.length ( sr * duration )
                assert len(track) == self.length

                # tfrecord file save
                feature = dict()
                feature['track'] = _float_feature(track)
                for label in self.label_names:
                    feature[label] = _int64_feature([labels_dict[label][i]])
                sample = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(sample.SerializeToString())
                writer.close()

        print("===== %s Dataset ready!" % self.split)
        # tensorflow dataset from tfrecord files
        tfrecord_dataset = tf.data.TFRecordDataset(save_paths)
        # tfrecord data parsing
        dataset = tfrecord_dataset.map(lambda x: parse_record(x, self.length, self.label_names), num_parallel_calls= tf.data.experimental.AUTOTUNE)
        # signal process with apply
        dataset = dataset.map(lambda x, y: (tf_Signal_Process(x, first_axis_is_batch=False, method=self.signal_process, sr=self.sr), y))
        return dataset

    def get_audio_path(self, audio_dir, track_id):
        return os.path.join(audio_dir, *track_id.split('/')).replace('mp3','wav')

    def audio_load(self, filepath, mono=True, sr=22050, duration=29.0):
        x, _ = librosa.load(filepath, sr=sr, mono=mono, duration=duration)
        # repeat so that all audio samples are the same length
        if len(x) != int(sr * duration):
            repeated_x = x
            for i in range(int(sr * duration) // len(x)):
                repeated_x = np.concatenate((repeated_x, x))
            x = repeated_x[:int(sr * duration)]
        assert len(x) == int(sr * duration)
        return x

if __name__ == "__main__":
    train_loader = DataLoader(metadata_path='metadata.csv', audio_dir='music_dataset', data_dir='data', split='training', signal_process='tf_mfcc')
    train_ds = train_loader.dataset()
    for x, y in train_ds.batch(3).take(1):
        print('x shape [batch_size, feature_size, sequence_length] : ', x.shape)
        print('x data type : ', x.dtype)
        print('y shape [batch_size, num_labels] : ', y.shape)
        print('y data type : ', y.dtype)

    valid_loader = DataLoader(metadata_path='metadata.csv', audio_dir='music_dataset', data_dir='data', split='validation', signal_process='tf_mfcc')
    valid_ds = valid_loader.dataset()

    for x, y in valid_ds.batch(3).take(1):
        print('x shape [batch_size, feature_size, sequence_length] : ', x.shape)
        print('x data type : ', x.dtype)
        print('y shape [batch_size, num_labels] : ', y.shape)
        print('y data type : ', y.dtype)