import os
import shutil
import sys

import numpy as np


def train_test_split_image_classification(folder_name, export_dir, root_dir_folder):
    classes_dir = os.listdir(folder_name)
    test_ratio = 0.2
    for classes in classes_dir:
        try:
            os.makedirs(export_dir + "/train/" + classes)
        except FileExistsError:
            pass
        try:
            os.makedirs(export_dir + "/test/" + classes)
        except FileExistsError:
            pass
    for i in range(len(classes_dir)):
        print('root', root_dir_folder)
        src = root_dir_folder + '/' + classes_dir[i]
        print(src)
        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(
            np.array(allFileNames), [int(len(allFileNames) * (1 - test_ratio))]
        )
        train_FileNames = [src + "/" + name for name in train_FileNames.tolist()]
        test_FileNames = [src + "/" + name for name in test_FileNames.tolist()]
        for name in train_FileNames:
            one = shutil.copy(name, export_dir + "/train/" + classes_dir[i])
        for name in test_FileNames:
            two = shutil.copy(name, export_dir + "/test/" + classes_dir[i])
    return one, two


def train_test_split_object_detection(folder_name, export_dir, root_dir_folder):
    all_filenames = [x for x in os.listdir(root_dir_folder) if x != "ImageSets"]
    full_path = []
    for f in all_filenames:
        full_path.append(os.path.abspath(folder_name + "/" + f))
        test_ratio = 0.2
        classes_dir = ["train", "test"]
    for classes in classes_dir:
        os.makedirs(export_dir + "/" + classes + "/images")
        os.makedirs(export_dir + "/" + classes + "/annotations")
    for i in range(len(full_path)):
        allFileNames = os.listdir(full_path[i])
        allFileNames = sorted(allFileNames)
        train_FileNames, test_FileNames = np.split(
            np.array(allFileNames), [int(len(allFileNames) * (1 - test_ratio))]
        )
        train_FileNames = [
            full_path[i] + "/" + name for name in train_FileNames.tolist()
        ]
        test_FileNames = [full_path[i] + "/" + name for name in test_FileNames.tolist()]
        for name in train_FileNames:
            if name[-3:] == "xml":
                shutil.copy(name, export_dir + "/train/annotations")
            else:
                shutil.copy(name, export_dir + "/train/images")
        for name in test_FileNames:
            if name[-3:] == "xml":
                shutil.copy(name, export_dir + "/test/annotations")
            else:
                shutil.copy(name, export_dir + "/test/images")
