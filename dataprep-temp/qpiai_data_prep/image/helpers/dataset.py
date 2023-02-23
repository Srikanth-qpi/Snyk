import glob
import os
import shutil

import qpiai as ap
from validator_collection import checkers, validators


def download(dataset):
    if checkers.is_url(dataset):
        filename = ap.download(dataset)
        img_list = []
        image_list = []
        if os.path.splitext(filename)[1] == ".zip":
            filename_folder_tmp = ap.unzip(filename)
            dataset = os.path.abspath(filename_folder_tmp)
            for f in os.listdir(dataset):
                file_path = os.path.join(dataset, f)
                if os.path.splitext(file_path)[1] == ".jpg":
                    img_list.append(file_path)
                    image_list.append(f)
                else:
                    all_filenames = [x[0] for x in os.walk(file_path)]
                    for i in all_filenames:
                        for f in glob.glob(os.path.join(i, "*.jpg")):
                            img_list.append(f)
        elif not os.path.splitext(filename)[1] == "":
            dataset = os.path.abspath(filename)
            img_list = [dataset]
            image_list = [dataset.split("/")[-1]]
        elif os.path.splitext(dataset)[1] == ".zip":
            filename_folder_tmp = ap.unzip(dataset)
            dataset = os.path.abspath(filename_folder_tmp)
            img_list = []
            image_list = []
            for f in os.listdir(dataset):
                file_path = os.path.join(dataset, f)
                img_list.append(file_path)
                image_list.append(f)
        elif not os.path.splitext(dataset)[1] == "":
            img_list = [dataset]
            image_list = [dataset.split("/")[-1]]
        else:
            if dataset[-1] == "/":
                dataset = dataset[:-1]
                img_list = []
                image_list = []
                for f in os.listdir(dataset):
                    file_path = os.path.join(dataset, f)
                    img_list.append(file_path)
                    image_list.append(f)
    return img_list, image_list


def save(folder_name, export_dir):
    root_dir_folder = os.path.abspath(folder_name)
    all_filenames = [x[0] for x in os.walk(folder_name)]
    all_filenames = all_filenames[1:]
    full_path = []
    for f in all_filenames:
        full_path.append(os.path.abspath(f))
        classes_dir = []
    for files in all_filenames:
        classes_dir.append(files.split("/")[-1])
    for classes in classes_dir:
        os.makedirs(export_dir + folder_name + "/" + classes)
    for each_image in os.listdir("Data_prep_images/"):
        print(each_image)
        path = os.path.abspath("Data_prep_images/" + each_image)
        print(path)
        ls = os.path.splitext(path)[0]
        ls = ls.split("/")
        ls = ls[-1]
        ls = ls.split("_")
        rs = ls[0]
        for i in classes_dir:
            if i == rs:
                src = os.path.abspath(
                    os.path.abspath("Data_prep_images/") + "/" + each_image
                )
                des = os.path.abspath(
                    os.path.abspath("Data_prep_images./") + "/" + folder_name + "/" + i
                )
                out = shutil.copy(src, des)
            else:
                out = None
    shutil.rmtree("Data_prep_images", ignore_errors=True)
    return out
