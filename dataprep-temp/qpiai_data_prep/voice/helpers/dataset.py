import os

import librosa
import numpy as np
from validator_collection import checkers

from qpiai_data_prep.files import download, unzip


def voice_download(dataset):
    if checkers.is_url(dataset):
        file_name = download(dataset)
        if os.path.splitext(file_name)[1] == ".zip":
            audio_file = unzip(file_name)
            x, sample_rate = librosa.load(audio_file, res_type="kaiser_fast")
        else:
            x, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    else:
        if os.path.splitext(dataset)[1] == ".zip":
            filename = download(dataset)
            filename_folder_tmp = unzip(filename)
            audio_file = os.path.abspath(filename_folder_tmp)
            x, sample_rate = librosa.load(audio_file, res_type="kaiser_fast")
        else:
            x, sample_rate = librosa.load(dataset, res_type="kaiser_fast")
    return x, sample_rate
