from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from fnmatch import fnmatch
import librosa, librosa.display
import numpy as np
import cv2
import multiprocessing
import parf
from multiprocessing import Pool
from itertools import repeat
from functools import partial

def par_create_mel(new_num, filenameslist, img_size, label_folder):
    """Parralel function to create Mel spectrograms """

    signal, sr = librosa.load(filenameslist[new_num])

    hop_length = int(len(signal) / img_size + 1)
    n_fft = 4096
    n_mels = img_size
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length,
                                                n_fft=n_fft, n_mels=n_mels)

    power_to_db = librosa.power_to_db(np.abs(mel_signal), ref=np.max)
    power_to_db = np.flip(power_to_db, axis=0)
    fig = scale_minmax(power_to_db, 0, 255).astype(np.uint8)
    img_filename = os.path.join(label_folder, str(new_num) + '.jpg')
    cv2.imwrite(img_filename, fig)

def par_preprocess_audio(new_num, filenameslist, save_folder, label, min_silence_len=300,
                         standart_length=5000, audio_pattern='*.wav'):
    """ Parralel function to preprocess audio - cut silent places, normalize length to 5 sec, save as wav"""

    # reading from audio file
    filename = filenameslist[new_num]
    format = 'm4a' if audio_pattern == '*.m4a' else 'wav'
    sound = AudioSegment.from_file(filename, format=format)

    # splitting audio files to cut silence and restore again
    res_audio = -1
    audio_chunks = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=-40)
    for i, chunk in enumerate(audio_chunks):
        if i == 0:
            res_audio = chunk
        else:
            res_audio = res_audio.append(chunk)

    # normalize length of the audio and save
    if (res_audio != -1) and (res_audio.duration_seconds < standart_length):
        while res_audio.duration_seconds < standart_length:
            res_audio = res_audio.append(res_audio)

        new_filename = os.path.join(save_folder, label, str(new_num) + '.wav')
        res_audio[0:standart_length].export(new_filename, format='wav')

    elif (res_audio != -1) and (res_audio.duration_seconds > standart_length):
        res_audio = res_audio.append(res_audio)
        new_filename1 = os.path.join(save_folder, label, str(new_num) + 'a' + '.wav')
        new_filename2 = os.path.join(save_folder, label, str(new_num) + 'b' + '.wav')
        res_audio[0:standart_length].export(new_filename1, format='wav')
        res_audio[standart_length:(2 * standart_length)].export(new_filename2, format='wav')

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

