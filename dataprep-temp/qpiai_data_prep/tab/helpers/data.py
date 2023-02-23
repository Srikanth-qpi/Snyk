import os

import pandas as pd
from validator_collection import checkers

from qpiai_data_prep.files import download, unzip


class data:
    def __init__(self):
        pass

    def download(dataset, **kwargs):
        if checkers.is_url(dataset):
            filename = download(dataset)
            if os.path.splitext(filename)[1] == ".zip":
                file_csv = unzip(filename)
                df = pd.read_csv(file_csv, sep=kwargs["dataframe_delimiter"])
            else:
                df = pd.read_csv(filename, sep=kwargs["dataframe_delimiter"])
        else:
            if os.path.splitext(dataset)[1] == ".zip":
                filename = download(dataset)
                filename_folder_tmp = unzip(filename)
                file_csv = os.path.abspath(filename_folder_tmp)
                df = pd.read_csv(file_csv, sep=kwargs["dataframe_delimiter"])
            else:
                df = pd.read_csv(dataset, sep=kwargs["dataframe_delimiter"])

        return df
