import os
import shutil

from validator_collection import checkers

from qpiai_data_prep.files import download, unzip


def video_download(dataset):
    if checkers.is_url(dataset):
        filename = download(dataset)
        vid_list = []
        video_list = []
        if os.path.splitext(filename)[1] == ".zip":
            filename_folder_tmp = unzip(filename)
            dataset = os.path.abspath(filename_folder_tmp)
            for f in os.listdir(dataset):
                file_path = os.path.join(dataset, f)
                vid_list.append(file_path)
                video_list.append(f)
        elif not os.path.splitext(filename)[1] == "":
            dataset = os.path.abspath(filename)
            vid_list = [dataset]
            video_list = [dataset.split("/")[-1]]
    elif os.path.splitext(dataset)[1] == ".zip":
        filename_folder_tmp = unzip(dataset)
        dataset = os.path.abspath(filename_folder_tmp)
        vid_list = []
        video_list = []
        for f in os.listdir(dataset):
            file_path = os.path.join(dataset, f)
            vid_list.append(file_path)
            video_list.append(f)

    elif not os.path.splitext(dataset)[1] == "":
        vid_list = [dataset]
        video_list = [dataset.split("/")[-1]]
    else:
        if dataset[-1] == "/":
            dataset = dataset[:-1]
            vid_list = []
            video_list = []
            for f in os.listdir(dataset):
                file_path = dataset + "/" + f
                vid_list.append(file_path)
                video_list.append(f)

    return vid_list, video_list
