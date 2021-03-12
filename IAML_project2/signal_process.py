import numpy as np
import tensorflow as tf
import librosa

# tf_Signal_Process have to be used for dataset.map function
@tf.function
def tf_Signal_Process(audio_samples, first_axis_is_batch=False, sr=22050, method='stft'):
    tf_float = tf.py_function(
        Signal_Process,
        [audio_samples, first_axis_is_batch, sr, method],
        tf.float32
    )
    return tf_float

def Signal_Process(audio_samples, first_axis_is_batch=False, sr=22050, method='stft'):
    """
    :param audio_samples: sampled raw audio input (tf.Tensor)
    :param first_axis_is_batch: first axis means batch, default = False
    :param sr: sampling rate
    :param method: signal process methods
    :return: signal_processed output [feature_size, sequence_length]
    """

    # TODO: define your signal process method with various functions and hyper parameters
    if method == 'your_own_way':
        stfts = tf.signal.stft(audio_samples, frame_length=2048, frame_step=512, fft_length=2048, pad_end=True)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr,
                                                                            lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            f1 = list()
            f2 = list()
            for i in range(len(audio_samples)):
                spectral_centroids = np.abs(librosa.feature.spectral_centroid(audio_samples[i], sr=sr)[0])
                spectral_rolloff = np.abs(librosa.feature.spectral_rolloff(audio_samples[i], sr=sr)[0])
                zero_crossing_rate = np.abs(librosa.feature.zero_crossing_rate(audio_samples[i], pad=False)[0])

                f.append(np.abs(spectral_centroids))
                f1.append(np.abs(spectral_rolloff))
                f2.append(np.abs(zero_crossing_rate))
            # spectral_centroids
            f = tf.convert_to_tensor(f, dtype=tf.float32)
            f = tf.reshape(f, (f.shape[0], f.shape[1], 1))
            spectral_centroids = tf.transpose(f, perm=[0, 2, 1])

            # spectral_rolloff
            f1 = tf.convert_to_tensor(f1, dtype=tf.float32)
            f1 = tf.reshape(f1, (f1.shape[0], f1.shape[1], 1))
            spectral_rolloff = tf.transpose(f1, perm=[0, 2, 1])

            # zero_crossing_rate
            f2 = tf.convert_to_tensor(f2, dtype=tf.float32)
            f2 = tf.reshape(f2, (f2.shape[0], f2.shape[1], 1))
            zero_crossing_rate = tf.transpose(f2, perm=[0, 2, 1])

            # log_mel_spectrograms
            log_mel_spectrograms = tf.transpose(log_mel_spectrograms, perm=[0, 2, 1])
            spectral_mel = tf.concat([log_mel_spectrograms, zero_crossing_rate, spectral_centroids, spectral_rolloff],
                                     axis=1)
            # print("spectral_mel.shape: ", spectral_mel.shape)

            return spectral_mel
        else:
            # zero-crossing rate
            x = np.abs(librosa.feature.zero_crossing_rate(audio_samples, pad=False)[0])
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            x = tf.reshape(x, (x.shape[0], 1))
            zero_crossing_rate = tf.transpose(x, perm=[1, 0])
            # print("zero_crossing_rate.shape: ", zero_crossing_rate.shape)

            # spectral centroid
            x1 = np.abs(librosa.feature.spectral_centroid(audio_samples, sr=sr)[0])
            x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
            x1 = tf.reshape(x1, (x1.shape[0], 1))
            spectral_centroids = tf.transpose(x1, perm=[1, 0])
            # print("spectral_centroids.shape: ", spectral_centroids.shape)

            # spectral Rolloff
            x2 = np.abs(librosa.feature.spectral_rolloff(audio_samples, sr=sr)[0])
            x2 = tf.convert_to_tensor(x2, dtype=tf.float32)
            x2 = tf.reshape(x2, (x2.shape[0], 1))
            spectral_rolloff = tf.transpose(x2, perm=[1, 0])
            # print("spectral_rolloff.shape: ", spectral_rolloff.shape)

            # log_mel_spectrograms
            log_mel_spectrograms = tf.transpose(log_mel_spectrograms, perm=[1, 0])
            # print("log_mel_spectrograms.shape: ", log_mel_spectrograms.shape)

            # concatenate
            spectral_mel = tf.concat([log_mel_spectrograms, zero_crossing_rate, spectral_centroids, spectral_rolloff],
                                     axis=0)
            # print("spectral_mel.shape: ", spectral_mel.shape)

            return spectral_mel

    elif method =='raw_audio':
        return audio_samples

    elif method == 'tf_stft':
        stfts = tf.signal.stft(audio_samples, frame_length=2048, frame_step=512, fft_length=2048, pad_end=True)
        stfts = tf.abs(stfts)
        if first_axis_is_batch:
            return tf.transpose(stfts, perm=[0, 2, 1])
        else:
            return tf.transpose(stfts, perm=[1, 0])

    elif method == 'tf_mel_spectrogram':
        stfts = tf.signal.stft(audio_samples, frame_length=2048, frame_step=512, fft_length=2048, pad_end=True)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr,
                                                                            lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        if first_axis_is_batch:
            return tf.transpose(mel_spectrograms, perm=[0, 2, 1])
        else:
            return tf.transpose(mel_spectrograms, perm=[1, 0])

    elif method == 'tf_log_mel_spectrogram':
        stfts = tf.signal.stft(audio_samples, frame_length=2048, frame_step=512, fft_length=2048, pad_end=True)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr,
                                                                            lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        if first_axis_is_batch:
            return tf.transpose(log_mel_spectrograms, perm=[0, 2, 1])
        else:
            return tf.transpose(log_mel_spectrograms, perm=[1, 0])

    elif method == 'tf_mfcc':
        stfts = tf.signal.stft(audio_samples, frame_length=2048, frame_step=512, fft_length=2048, pad_end=True)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr,
                                                                            lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :20]
        if first_axis_is_batch:
            return tf.transpose(mfccs, perm=[0, 2, 1])
        else:
            return tf.transpose(mfccs, perm=[1, 0])

    elif method == 'stft':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                f.append(np.abs(librosa.stft(audio_samples[i], n_fft=2048, hop_length=512)))
        else:
            f = np.abs(librosa.stft(audio_samples, n_fft=2048, hop_length=512))
        return tf.convert_to_tensor(f, dtype=tf.float32)

    elif method == 'cqt':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                f.append(np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=512, bins_per_octave=12, n_bins=7 * 12)))
        else:
            f = np.abs(librosa.cqt(audio_samples, sr=float(sr), hop_length=512, bins_per_octave=12, n_bins=7 * 12))
        return tf.convert_to_tensor(f, dtype=tf.float32)

    elif method == 'chroma_cqt':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                cqt = np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
                f.append(librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7))
        else:
            cqt = np.abs(
                librosa.cqt(audio_samples, sr=float(sr), hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
            f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        return tf.convert_to_tensor(f, dtype=tf.float32)

    elif method == 'chroma_cens':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                cqt = np.abs(librosa.cqt(audio_samples[i], sr=float(sr), hop_length=512, bins_per_octave=12, n_bins=7 * 12,
                                         tuning=None))
                f.append(librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7))
        else:
            cqt = np.abs(
                librosa.cqt(audio_samples, sr=float(sr), hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
            f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        return tf.convert_to_tensor(f, dtype=tf.float32)

    elif method == 'chroma_stft':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                stft = np.abs(librosa.stft(audio_samples[i], n_fft=2048, hop_length=512))
                f.append(librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12))
        else:
            stft = np.abs(librosa.stft(audio_samples, n_fft=2048, hop_length=512))
            f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
        return tf.convert_to_tensor(f, dtype=tf.float32)

    elif method == 'rms':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                stft = np.abs(librosa.stft(audio_samples[i], n_fft=2048, hop_length=512))
                f.append(librosa.feature.rms(S=stft))
        else:
            stft = np.abs(librosa.stft(audio_samples, n_fft=2048, hop_length=512))
            f = librosa.feature.rms(S=stft)
        return tf.convert_to_tensor(f, dtype=tf.float32)

    elif method == 'mel_spectrogram':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                stft = np.abs(librosa.stft(audio_samples[i], n_fft=2048, hop_length=512))
                f.append(librosa.feature.melspectrogram(S=stft ** 2, sr=sr))
        else:
            stft = np.abs(librosa.stft(audio_samples, n_fft=2048, hop_length=512))
            f = librosa.feature.melspectrogram(S=stft ** 2, sr=sr)
        return tf.convert_to_tensor(f, dtype=tf.float32)

    elif method == 'mfcc':
        audio_samples = audio_samples.numpy()
        if first_axis_is_batch:
            f = list()
            for i in range(len(audio_samples)):
                stft = np.abs(librosa.stft(audio_samples[i], n_fft=2048, hop_length=512))
                mel_spectrogram = librosa.feature.melspectrogram(S=stft ** 2, sr=sr)
                f.append(librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=20))
        else:
            stft = np.abs(librosa.stft(audio_samples, n_fft=2048, hop_length=512))
            mel_spectrogram = librosa.feature.melspectrogram(S=stft ** 2, sr=sr)
            f = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=20)
        return tf.convert_to_tensor(f, dtype=tf.float32)

    else:
        raise NotImplementedError

if __name__ == "__main__":
    sr = 22050
    sec = 29.0

    fake_audio_samples = tf.tanh(tf.random.normal([int(sr * sec)]))
    print('%.1f sec and sampling rate %d fake audio : %s' % (sec, sr, fake_audio_samples.shape))

    raw_audio = Signal_Process(fake_audio_samples, first_axis_is_batch=False, sr=sr, method='raw_audio')
    print('raw audio : %s' % raw_audio.shape)

    tf_stft = Signal_Process(fake_audio_samples, first_axis_is_batch=False, sr=sr, method='tf_stft')
    print('stft with tensorflow : %s' % tf_stft.shape)

    tf_mfcc = Signal_Process(fake_audio_samples, first_axis_is_batch=False, sr=sr, method='tf_mfcc')
    print('mfcc with tensorflow : %s' % tf_mfcc.shape)

    cqt = Signal_Process(fake_audio_samples, first_axis_is_batch=False, sr=sr, method='cqt')
    print('constant q transform with librosa : %s' % cqt.shape)

    mel_spectrogram = Signal_Process(fake_audio_samples, first_axis_is_batch=False, sr=sr, method='mel_spectrogram')
    print('mel spectrogram with librosa : %s' % mel_spectrogram.shape)

    batch_size = 4

    batch_audio_samples = tf.tanh(tf.random.normal([batch_size, int(sr*sec)]))
    print('%.1f sec and sampling rate %d, 4 batch audio samples : %s' % (sec, sr, batch_audio_samples.shape))

    tf_mfcc = Signal_Process(batch_audio_samples, first_axis_is_batch=True, sr=sr, method='tf_mfcc')
    print('mfcc with tensorflow for batch input : %s' % tf_mfcc.shape)

    cqt = Signal_Process(batch_audio_samples, first_axis_is_batch=True, sr=sr, method='cqt')
    print('constant q transform with librosa for batch input : %s' % cqt.shape)

    print('\ndataset applied map function (tf_Signal_Process : mfcc) ')
    dataset = tf.data.Dataset.from_tensor_slices(batch_audio_samples)
    dataset = dataset.map(lambda x: tf_Signal_Process(fake_audio_samples, first_axis_is_batch=False, sr=sr, method='mfcc'))
    for x in dataset.batch(2).take(1):
        print('x shape [batch_size, feature_size, sequence_length] : ', x.shape)
        print('x data type : ', x.dtype)