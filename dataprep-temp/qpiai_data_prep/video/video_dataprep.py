import json
import math as m
import os
import shutil
import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imshow
from validator_collection import checkers, validators

from qpiai_data_prep.video.helpers.dataset import *
from qpiai_data_prep.video.helpers.utils import *
from temp.db_info import db

PATH = list()


def video_dataprep(dataset, target_device, num_device, data_prepcmd):

    db.update_progress(progress=5)
    PATH = list()
    if os.path.exists("Data_prep_videos/"):
        shutil.rmtree("Data_prep_videos", ignore_errors=True)
    os.mkdir("Data_prep_videos")
    export_dir = os.path.join(os.getcwd(), "Data_prep_videos")

    db.update_progress(progress=15)
    vid_list, video_list = video_download(dataset)
    db.update_progress(progress=35)

    for video in vid_list:
        name = Path(video).name
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        frame_height, frame_widht, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            os.path.join(export_dir, "output.avi"),
            fourcc,
            20.0,
            (frame_height, frame_widht),
        )
        fgbg = cv2.createBackgroundSubtractorMOG2()
        db.update_progress(progress=50)
        while True:
            ret, frame = cap.read()
            if ret == True:
                if data_prepcmd == "EdgeDetection":
                    Vid_ops.edge_detection(frame, out)
                elif data_prepcmd == "Enhancement":
                    Vid_ops.enhancement(frame, out)
                elif data_prepcmd == "Blur":
                    Vid_ops.blur(frame, out)
                elif data_prepcmd == "CornerDetection":
                    Vid_ops.corner_detection(frame, out)
                elif data_prepcmd == "rbc":
                    Vid_ops.rbc(frame, out)
                elif data_prepcmd == "shift_scale_rotate":
                    Vid_ops.shift_scale_rotate(frame, out)
                elif data_prepcmd == "color_jitter":
                    Vid_ops.color_jitter(frame, out)
                elif data_prepcmd == "rgb_shift":
                    Vid_ops.rgb_shift(frame, out)
                elif data_prepcmd == "channel_shuffle":
                    Vid_ops.channel_shuffle(frame, out)
                elif data_prepcmd == "random_fog":
                    Vid_ops.random_fog(frame, out)
                elif data_prepcmd == "random_rain":
                    Vid_ops.random_rain(frame, out)
                elif data_prepcmd == "random_shadow":
                    Vid_ops.random_shadow(frame, out)
                elif data_prepcmd == "random_sunflare":
                    Vid_ops.random_sunflare(frame, out)
                elif data_prepcmd == "random_snow":
                    Vid_ops.random_snow(frame, out)
                else:
                    return "Please provide Valid Input Dataprep Command"

            else:
                break

    for each_video in os.listdir("Data_prep_videos"):
        full_path = os.path.abspath("Data_prep_videos/")
        PATH.append(os.path.join(full_path, each_video))

    checkpoint = dict({"dataPrepOutput": PATH})
    print(json.dumps(checkpoint))
    return checkpoint

    # cv2.destroyAllWindows()


# video_dataprep('https://dataprepfiles.s3.amazonaws.com/samplevideo.mp4','cpu',1,'random_snow')
