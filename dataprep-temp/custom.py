import os
import shutil
import subprocess
from qpiai_data_prep.files import download, unzip
from qpiai_data_prep.image.helpers.data_set import *
import pandas as pd
import importlib

def run(zipfile, dataset, datatype):
    file_name = download(zipfile)
    data = download(dataset)
    file_name = unzip(file_name)
    paths = os.path.abspath(file_name)
    path = paths + '/' + 'requirements.txt'
    path1 = paths + '/' + 'sample.py'
    path2 = paths + '/' + 'conf_args.py'
    shutil.copy(path, os.getcwd())
    shutil.copy(path1, os.getcwd())
    shutil.copy(path2, os.getcwd())
    subprocess.Popen('pip3 install -r requirements.txt', shell=True)
    from sample import func
    from conf_args import config_dict
    if datatype == 'custom_tab':
        data = download(dataset)
        df = pd.read_csv(data)
        output= func(df=df, **config_dict)
    elif datatype == 'custom_image':
        if os.path.exists(config_dict['folder']):
            shutil.rmtree(config_dict['folder'], ignore_errors=True)
        img_list, image_list, _, _ = image_download(dataset)
        output = func(img_list, **config_dict)
    checkpoint = {"dataPrepOutput": output}
    return checkpoint



#run('https://dataprepfiles.s3.amazonaws.com/code1.zip', 'https://dataprepfiles.s3.amazonaws.com/CardioGoodFitness.csv', 'tab')