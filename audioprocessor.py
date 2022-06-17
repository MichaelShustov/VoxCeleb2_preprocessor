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
from datetime import datetime


class WavPreProc:
    """ Preprocessor for audio files - cut silent places, convert to Mel-spectrograms, saves as jpeg
    """
    def __init__(self, big_archive_folder, train_folder, val_folder):
        """
        Constructor
        :param big_archive_folder: Folder of initial dataset
        :param train_folder: folder to save train dataset
        :param val_folder: folder to save validation dataset
        """

        self._big_archive_folder = big_archive_folder
        self._train_folder = train_folder
        self._val_folder = val_folder

        # create folders to save validation and train datasets
        try:
            os.mkdir(self._train_folder)
            os.mkdir(self._val_folder)
        except:
            pass

    def get_archive_folder(self):
        """ Getter for main dataset folder name"""
        return self._big_archive_folder

    def get_val_folder(self):
        """ Getter for validation dataset folder name"""
        return self._val_folder

    def get_train_folder(self):
        """ Getter for train dataset folder name"""
        return self._train_folder

    def read_subfolders_names(self,folder):
        """
        Returns names of subfolders in the folder
        """

        for path, subfolders, files in os.walk(folder):
            return subfolders

    def read_file_names(self, folder, pattern='*.wav'):
        """
         Returns list of all patterned files in the folder
        """

        filenameslist = list()

        for path, subfolders, files in os.walk(folder):
            for name in files:
                if fnmatch(name, pattern):
                    filenameslist.append(os.path.join(path, name))

        return filenameslist

    def preprocess_audio(self, filenameslist, save_folder, label, min_silence_len=300,
                         standart_length=5000, audio_pattern='*.wav'):
        """
        Reads file (filename) cut silence places and saves it into save_folder/label/new_num.wav
        """

        num_list = list(range(len(filenameslist)))
        pool = multiprocessing.Pool(processes=4)
        pool.starmap(parf.par_preprocess_audio, zip(num_list, repeat(filenameslist),
                                                    repeat(save_folder), repeat(label),
                                                    repeat(min_silence_len), repeat(standart_length),
                                                    repeat(audio_pattern)))

        pool.close()
        pool.join()
        print('done')


    def create_mels(self, folder, img_size=160, audio_pattern='*.wav'):
        """
        Creates Mel-spectrograms for all labels in folder and saves as jpeg
        :param folder: train folder or validation folder
        :param img_size: size of the image (square-shaped)
        :param audio_pattern: '*.wav' or '*.m4a'
        :return:
        """

        labels = pp.read_subfolders_names(folder)

        for label in labels:
            print('Processing Mels for label ' + label + ' in a folder ' + folder)
            label_folder = os.path.join(folder, label)
            filenameslist = pp.read_file_names(label_folder, pattern=audio_pattern)

            num_list = list(range(len(filenameslist)))

            pool = multiprocessing.Pool(processes=4)
            pool.starmap(parf.par_create_mel, zip(num_list, repeat(filenameslist),
                                                  repeat(img_size), repeat(label_folder)))

            pool.close()
            pool.join()
            print('done')


    def preprocess_n_audio_labels(self, labels_num, min_silence_len=300, val_portion=0.1, audio_pattern='*.wav'):
        """
        Makes preprocesing for audio files - distributes into train-validation folders, cut quite places
        :param labels_num: how many labels to process. The task was for 1000
        :param min_silence_len: minimal length in ms of silent places in audio, which are removed
        :param val_portion: portion of data taken to validation dataset
        :param audio_pattern: '*.wav' or '*.m4a'
        :return:
        """

        labels = self.read_subfolders_names(self._big_archive_folder)
        labels_num = len(labels) if (labels_num>len(labels)) else labels_num

        for i in range(labels_num):
            label_folder = os.path.join(self._big_archive_folder, labels[i])
            label_sub = self.read_subfolders_names(label_folder)
            subfolders_num = len(label_sub)
            val_num = int(subfolders_num * val_portion)
            train_num = subfolders_num - val_num

            # make list of validation files
            val_filenames = list()
            for j in range(val_num):
                curr_folder = os.path.join(self._big_archive_folder, labels[i],label_sub[j])
                val_filenames = val_filenames + self.read_file_names(curr_folder,pattern=audio_pattern)

            # make list of train files
            train_filenames = list()
            for j in range(train_num):
                j = j +val_num
                curr_folder = os.path.join(self._big_archive_folder, labels[i], label_sub[j])
                train_filenames = train_filenames + self.read_file_names(curr_folder,pattern=audio_pattern)

            # create folders to save validation and train datasets
            try:
                os.mkdir(os.path.join(self._train_folder, labels[i]))
                os.mkdir(os.path.join(self._val_folder, labels[i]))
            except:
                pass

            # run main audio process functions
            print('PreProcessing audio for validation label:'+str(labels[i]))
            self.preprocess_audio(val_filenames, self._val_folder,labels[i], min_silence_len = min_silence_len,
                                    standart_length = 5000, audio_pattern=audio_pattern)
            print('PreProcessing audio for train label:' + str(labels[i]))
            self.preprocess_audio(train_filenames, self._train_folder, labels[i], min_silence_len=min_silence_len,
                                  standart_length=5000, audio_pattern=audio_pattern)


if __name__ == '__main__':
    pp = WavPreProc('archive', 'train', 'validation')
    start = datetime.now()
    pp.preprocess_n_audio_labels(20, min_silence_len=500, val_portion=0.2, audio_pattern='*.wav')
    pp.create_mels('train', img_size=160, audio_pattern='*.wav')
    pp.create_mels('validation', img_size=160, audio_pattern='*.wav')
    print(datetime.now() - start)








