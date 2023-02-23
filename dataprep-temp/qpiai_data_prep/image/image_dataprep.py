import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from validator_collection import checkers, validators

from qpiai_data_prep.files import download, unzip
from qpiai_data_prep.image.helpers.augmentations import *
from qpiai_data_prep.image.helpers.data_set import *
from qpiai_data_prep.image.helpers.train_test import *
from qpiai_data_prep.image.helpers.utils import *
from temp.db_info import db

def image_dataprep(dataset, target_device, num_device, data_prepcmd, **kwargs):
    PATH = list()

    if os.path.exists("Data_prep_images"):
        shutil.rmtree("Data_prep_images", ignore_errors=True)

    os.mkdir("Data_prep_images")

    export_dir = os.path.join(os.getcwd(), "Data_prep_images")

    flag = 1
    splitlist = [
        "train_test_split_image_classification",
        "train_test_split_object_detection",
    ]
    if data_prepcmd in splitlist:
        db.update_progress(progress=5)
        flag = 0
        if checkers.is_url(dataset):
            db.update_progress(progress=10)
            filename = download(dataset)
            db.update_progress(progress=30)
            if os.path.splitext(filename)[1] == ".zip":
                folder_name = unzip(filename)
        elif os.path.splitext(dataset)[1] == ".zip":
            folder_name = unzip(dataset)
            db.update_progress(progress=30)
        root_dir_folder = os.path.abspath(folder_name)
        db.update_progress(progress=50)
        if data_prepcmd == "train_test_split_image_classification":
            db.update_progress(progress=70)
            train_test_split_image_classification(
                folder_name, export_dir, root_dir_folder
            )

        if data_prepcmd == "train_test_split_object_detection":
            db.update_progress(progress=70)
            train_test_split_object_detection(folder_name, export_dir, root_dir_folder)
    else:
        db.update_progress(progress=5)
        img_list, image_list, filename, folder_name = image_download(dataset)
        db.update_progress(progress=20)
        tot = len(img_list)
        count=0
        for i in img_list:
            count=count+1
            if count/tot == 0.25:
                db.update_progress(progress=45)
            elif count/tot == 0.5:
                db.update_progress(progress=55)
            elif count/tot == 0.75:
                db.update_progress(progress=70)
            elif count/tot == 1:
                db.update_progress(progress=80)
            else:
                pass
            name = Path(i).name
            image = cv2.imread(i)
            if data_prepcmd == "enhancement":
                Img_ops.enhancement(image, name)
            elif data_prepcmd == "edge_detection":
                Img_ops.edge_detection(image, name)
            elif data_prepcmd == "corner_detection":
                Img_ops.corner_detection(image, name)
            elif data_prepcmd == "grayscale":
                Img_ops.gray_scale(image, name)
            elif data_prepcmd == "blur":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "vertical_flip":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "horizontal_flip":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "sharpen":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "channel_shuffle":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "channel_dropout":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "clahe":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "color_jitter":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "downscale":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "sepia":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "fancy_pca":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "gaussian_blur":  # only takes odd blur parameters
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "equalize":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "rgb_shift":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_brightness":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "median_blur":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_contrast":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "rbc":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "optical_distortion":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "flip":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "gauss_noise":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "glass_blur":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "emboss":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "grid_distortion":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "hue_sat_value":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "elastic_transform":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "posterize":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_fog":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_rain":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_shadow":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_sunflare":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_snow":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "cutout":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "transpose":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "std_normalize":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "normalize":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "multi_noise":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_crop":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "crop":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "center_crop":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "pad_if_needed":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "motion_blur":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "jpeg_compression":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "iso_noise":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "invert_img":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "resize":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "random_scale":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "rotate":
                Img_ops.base_transform(data_prepcmd, image, name, **kwargs)
            elif data_prepcmd == "shift_scale_rotate":  # augmentation
                Img_aug.base_func(data_prepcmd, image, name, **kwargs)

    if flag == 1:
        if checkers.is_url(dataset):
            if os.path.splitext(filename)[1] == ".zip":
                f = os.listdir(folder_name)
                if (
                    f[0][-4:] == ".jpg"
                    or f[0][-4:] == ".png"
                    or f[0][-4:] == ".JPG"
                    or f[0][-5:] == ".jpeg"
                    or f[0][-5:] == ".tiff"
                ):
                    if os.path.exists("dataset"):
                        shutil.rmtree("dataset", ignore_errors=True)
                    os.makedirs('dataset')
                    shutil.move(os.path.abspath('Data_prep_images'), os.path.abspath('dataset'))
                    shutil.make_archive("dataset", "zip", "dataset")
                    route = os.path.abspath("dataset.zip")
                    PATH.append(route)
                    checkpoint = dict({"dataPrepOutput": PATH})
                    return checkpoint
                else:
                    if os.path.exists("Data_prep_images/"):
                        shutil.rmtree("Data_prep_images.", ignore_errors=True)
                    one = save(folder_name)
                    path = os.path.abspath("Data_prep_images.zip")
                    #shutil.copy("Data_prep_images/dataset.zip", os.getcwd())
                    PATH.append(path)
                    checkpoint = dict({"dataPrepOutput": PATH})
                    return checkpoint

        elif os.path.splitext(dataset)[1] == ".zip":
            f = os.listdir(folder_name)
            if (
                f[0][-4:] == ".jpg"
                or f[0][-4:] == ".png"
                or f[0][-4:] == ".JPG"
                or f[0][-5:] == ".jpeg"
                or f[0][-5:] == ".tiff"
            ):
                if os.path.exists("dataset"):
                        shutil.rmtree("dataset", ignore_errors=True)
                os.makedirs('dataset')
                shutil.move(os.path.abspath('Data_prep_images'), os.path.abspath('dataset'))
                shutil.make_archive("dataset", "zip", "dataset")
                route = os.path.abspath("dataset.zip")
                PATH.append(route)
                checkpoint = dict({"dataPrepOutput": PATH})
                return checkpoint
            else:
                if os.path.exists("Data_prep_images/"):
                    shutil.rmtree("Data_prep_images.", ignore_errors=True)
                    one = save(folder_name)
                    path = os.path.abspath("Data_prep_images.zip")
                    #shutil.copy("Data_prep_images/dataset.zip", os.getcwd())
                    PATH.append(path)
                    checkpoint = dict({"dataPrepOutput": PATH})
                    return checkpoint

        elif os.path.splitext(dataset)[1] != ".zip":
            for each_image in os.listdir("Data_prep_images"):
                full_path = os.path.abspath("Data_prep_images/")
                PATH.append(os.path.join(full_path, each_image))
            checkpoint = dict({"dataPrepOutput": PATH})
            return checkpoint
    else:
        if os.path.exists("dataset"):
            shutil.rmtree("dataset", ignore_errors=True)
        os.makedirs('dataset')
        shutil.move(os.path.abspath('Data_prep_images'), os.path.abspath('dataset'))
        shutil.make_archive("dataset", "zip", "dataset")
        #shutil.make_archive("Data_prep_images", "zip", "Data_prep_images")
        return {"dataPrepOutput": os.path.abspath("dataset.zip")}


# image_dataprep('https://dataprepfiles.s3.amazonaws.com/image.jpg','cpu',1,'shift_scale_rotate')
# image_dataprep('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip','cpu',1,'train_test_split_object_detection')
# image_dataprep('https://qpiaidataset.s3.amazonaws.com/shoppie_data.zip','cpu',1,'train_test_split_image_classification')
# image_dataprep('https://qpiaidataset.s3.amazonaws.com/shoppie_data.zip','cpu',1,'edge_detection')

# image_dataprep('sampics.zip','cpu',1,'edge_detection')
