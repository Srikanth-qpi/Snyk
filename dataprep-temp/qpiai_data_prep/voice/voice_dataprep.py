import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from validator_collection import checkers

from qpiai_data_prep.voice.helpers.dataset import *
from qpiai_data_prep.voice.helpers.utils import *
from temp.db_info import db

def voice_dataprep(
    dataset,
    target_device,
    num_device,
    data_prepcmd,
    n_fft=2048,
    hop_length=512,
    num_segments=5,
    n_mfcc=40,
):
    db.update_progress(progress=5)
    x, sample_rate = voice_download(dataset)
    db.update_progress(progress=30)
    if data_prepcmd == "Waveform":
        filename = "waveform.png"
        db.update_progress(progress=60)
        voice_ops.waveform(x, sample_rate, filename)
    elif data_prepcmd == "Denoise":
        filename = "spec.png"
        db.update_progress(progress=60)
        voice_ops.denoise(x, sample_rate, filename)
    elif data_prepcmd == "MFCC":
        filename = "mfcc.csv"
        db.update_progress(progress=60)
        voice_ops.mfcc(x, sample_rate, filename)
    elif data_prepcmd == "FastFourierTransform":
        filename = "fft.png"
        db.update_progress(progress=60)
        voice_ops.fft(x, sample_rate, filename)
    elif data_prepcmd == "ShortTimeFourierTransform":
        filename = "sft.png"
        db.update_progress(progress=60)
        voice_ops.sft(x, sample_rate, filename)
    elif data_prepcmd == "Beats_count":
        filename = "beats.csv"
        voice_ops.beats_count(x, sample_rate, filename)
    elif data_prepcmd == "noise_addition":
        filename = "noise_add.wav"
        db.update_progress(progress=60)
        voice_ops.noise_addition(x, sample_rate, filename)
    elif data_prepcmd == "shift_time":
        filename = "shift_add.wav"
        db.update_progress(progress=60)
        voice_ops.shift_time(x, sample_rate, filename)
    elif data_prepcmd == "stretch_time":
        filename = "stretch.wav"
        db.update_progress(progress=60)
        voice_ops.stretch_time(x, sample_rate, filename)
    elif data_prepcmd == "shift_pitch":
        filename = "shift.wav"
        db.update_progress(progress=60)
        voice_ops.shift_pitch(x, sample_rate, filename)
    else:
        return "Please Provide Valid Dataprep Command"

    # checkpoint = os.path.abspath(filename)
    PATH = list()
    PATH.append(os.path.abspath(filename))
    db.update_progress(progress=90)
    checkpoint = dict({"dataPrepOutput": PATH})
    print(json.dumps(checkpoint))
    return checkpoint
