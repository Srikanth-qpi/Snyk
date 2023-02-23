import os

import pandas as pd
from validator_collection import checkers

from qpiai_data_prep.files import download, unzip


def text_download(dataset):
    if checkers.is_url(dataset):
        file_name = download(dataset)
        if os.path.splitext(file_name)[1] == ".zip":
            text_file = unzip(file_name)
            f_le = open(text_file, "r")
        elif file_name.split(".")[-1] == "txt":
            f_le = open(file_name, "r", encoding='utf-8')
        elif file_name.split(".")[-1] == "csv":
            f_le = pd.read_csv(file_name)
        else:
            f_le = open(file_name, "r")
    else:
        if os.path.splitext(dataset)[1] == ".zip":
            filename = download(dataset)
            filename_folder_tmp = unzip(filename)
            text_file = os.path.abspath(filename_folder_tmp)
            f_le = open(text_file, "r")
        else:
            f_le = open(dataset, "r")
    return f_le, file_name
