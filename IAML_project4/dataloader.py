import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import pretty_midi

def get_midi_paths(data_path):
    '''
    :param data_path: data directory
    :return: list of piano rolls
    '''
    paths = []
    for path, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.midi'):
                paths.append(os.path.join(path, file))
    return paths

def midi_to_pianoroll(path):
    '''
    :param path: midi path
    :return: piano roll of shape [256, 128]
    '''
    pm = pretty_midi.PrettyMIDI(path)
    pianoroll = pm.get_piano_roll(fs=8)
    pianoroll = np.array(pianoroll, dtype=np.int32).T
    if pianoroll.shape[0] > 256:
        raise NotImplementedError
    elif pianoroll.shape[0] < 256:
        mask = 256 - pianoroll.shape[0]
        return np.pad(pianoroll, [(0, mask), (0, 0)], 'constant')
    else:
        return pianoroll

def save_pianoroll_to_midi(piano_roll, filename, fs=8, program=0):
    '''Convert a Piano Roll array into a midi file with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(frames, 128), dtype=int
        Piano roll of one instrument
    filename : save path
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    '''
    piano_roll = piano_roll.T
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    pm.write(filename)

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_record(example_proto, shape=[256,128]):
    # Parse record from TFRecord file
    feature_description = {
        'pianoroll': tf.io.FixedLenFeature([int(shape[0]*shape[1])], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    pianoroll = parsed['pianoroll']
    pianoroll = tf.reshape(pianoroll, shape)
    return pianoroll

class DataLoader(object):
    def __init__(self, audio_dir='music_dataset', data_dir='data'):
        '''
        :param audio_dir: midi file directory
        :param data_dir: tfrecord file directory
        '''
        self.audio_dir = audio_dir
        self.data_dir = data_dir

    def dataset(self):
        """
        :return: dataset with
                 midi data [sequence_length (256), pitch_size (128)]
        """
        # save loaded audio, label or load data
        midi_paths = get_midi_paths(self.audio_dir)
        save_paths = [os.path.join(self.data_dir, '%04d.tfrecord' % int(i[-9:-5])) for i in midi_paths]

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            print("===== Preparing pianoroll dataset! This could take a while...")

            for i in tqdm(range(len(midi_paths))):
                # tfrecord writer
                save_path = save_paths[i]
                writer = tf.io.TFRecordWriter(save_path)

                # midi file load
                pianoroll = midi_to_pianoroll(midi_paths[i])

                # pianoroll shape should be [sequence_length (256), pitch_size (128)]
                assert pianoroll.shape == (256, 128)

                # tfrecord file save
                feature = dict()
                feature['pianoroll'] = _float_feature(pianoroll.flatten()*1.0)
                sample = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(sample.SerializeToString())
                writer.close()

        print("===== Pianoroll Dataset ready!")
        # tensorflow dataset from tfrecord files
        tfrecord_dataset = tf.data.TFRecordDataset(save_paths)
        # tfrecord data parsing
        dataset = tfrecord_dataset.map(lambda x: parse_record(x), num_parallel_calls= tf.data.experimental.AUTOTUNE)
        print('===== Dataset loaded')
        return dataset

if __name__ == "__main__":
    data_loader = DataLoader(audio_dir='music_dataset', data_dir='data')
    dataset = data_loader.dataset()
    for x in dataset.batch(3).take(1):
        print('x shape [batch_size, sequence_length, pitch_size] : ', x.shape)
        print('x data type : ', x.dtype)