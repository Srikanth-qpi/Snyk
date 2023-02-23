import glob
import os
import shutil

from validator_collection import checkers, validators

from qpiai_data_prep.files import download, unzip


class Dataset():
    
    def __init__(self, dataset, dataset_format):
        self.dataset = dataset
        self.dataset_format = dataset_format


    def download(self):
        img_list = []
        image_list = []
        coco_list = []
        yolo_list = []
        if checkers.is_url(self.dataset):
            filename = download(self.dataset)
            if os.path.splitext(filename)[1] == ".zip":
                folder_name = unzip(filename)
                dataset = os.path.abspath(folder_name)
                for f in os.listdir(dataset):
                    file_path = os.path.join(dataset, f)
                    if os.path.splitext(file_path)[1] == ".jpg":
                        img_list.append(file_path)
                        image_list.append(f)
                    else:
                        print('getting into')
                        img_list, coco_list, yolo_list = self.extract_images(file_path, img_list=img_list, coco_list=coco_list, yolo_list=yolo_list)
            elif not os.path.splitext(filename)[1] == "":
                dataset = os.path.abspath(filename)
                img_list = [dataset]
                image_list = [dataset.split("/")[-1]]
        elif os.path.splitext(self.dataset)[1] == ".zip":
            filename = self.dataset
            # folder_name = os.path.splitext(dataset)[0]
            folder_name = unzip(self.dataset)
            if type(folder_name) is tuple:
                folder_name = os.path.splitext(self.dataset)[0]
                shutil.unpack_archive(self.dataset, folder_name)
            dataset = os.path.abspath(folder_name)
            for f in os.listdir(dataset):
                file_path = os.path.join(dataset, f)
                if (os.path.splitext(file_path)[1] in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.PNG', '.png' ]):
                    img_list.append(file_path)
                    image_list.append(f)
                else:
                    img_list, coco_list, yolo_list = self.extract_images(file_path, img_list=img_list, coco_list=coco_list, yolo_list=yolo_list)
        
        if self.dataset_format == 'coco':
            annot_list = coco_list
        elif self.dataset_format == 'yolo':
            annot_list = yolo_list
        elif self.dataset_format == 'classification':
            annot_list = []
        return img_list, image_list, annot_list, filename, folder_name

    def extract_images(self, file_path, img_list, coco_list, yolo_list):
        print('file path', file_path)
        all_filenames = [x[0] for x in os.walk(file_path)]
        print('all', all_filenames)
        for i in all_filenames:
            print('hahahah')
            for j in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.PNG', '*.png' ]:
                for f in glob.glob(os.path.join(i, j)):
                    img_list.append(f)
            for f in glob.glob(os.path.join(i, "*.json")):
                coco_list.append(f)
            for f in glob.glob(os.path.join(i, "*.txt")):
                yolo_list.append(f)
        return img_list, coco_list, yolo_list






# if __name__ == "__main__":
#     d = Dataset(dataset= '/home/ec2-user/dataprep/task_test-2022_11_16_13_48_35-coco 1.0.zip', dataset_format='coco')
#     img_list, image_list, annot_list, filename, folder_name = d.download()
#     print(img_list, image_list, annot_list)








# def image_download(dataset, dataset_format):
#     if dataset_format == "coco":
#         coco_list = []
#     raj = os.path.splitext(dataset)[1]
#     if checkers.is_url(dataset):
#         filename = download(dataset)
#         img_list = []
#         image_list = []
#         folder_name = unzip(filename)
#         if os.path.splitext(filename)[1] == ".zip":
#             dataset = os.path.abspath(folder_name)
#             for f in os.listdir(dataset):
#                 file_path = os.path.join(dataset, f)
#                 if os.path.splitext(file_path)[1] == ".jpg":
#                     img_list.append(file_path)
#                     image_list.append(f)
#                 else:
#                     all_filenames = [x[0] for x in os.walk(file_path)]
#                     for i in all_filenames:
#                         for f in glob.glob(os.path.join(i, "*.jpg")):
#                             img_list.append(f)
#                         for f in glob.glob(os.path.join(i, "*.json")):
#                             coco_list.append(f)
#         elif not os.path.splitext(filename)[1] == "":
#             dataset = os.path.abspath(filename)
#             img_list = [dataset]
#             image_list = [dataset.split("/")[-1]]
#     elif os.path.splitext(dataset)[1] == ".zip":
#         filename = dataset
#         # folder_name = os.path.splitext(dataset)[0]
#         folder_name = unzip(dataset)
#         if type(folder_name) is tuple:
#             folder_name = os.path.splitext(dataset)[0]
#             shutil.unpack_archive(dataset, folder_name)
#         # shutil.unpack_archive(dataset, folder_name)
#         dataset = os.path.abspath(folder_name)
#         img_list = []
#         image_list = []
#         for f in os.listdir(dataset):
#             file_path = os.path.join(dataset, f)
#             if (
#                 os.path.splitext(file_path)[1] == ".jpg"
#                 or os.path.splitext(file_path)[1] == ".JPG"
#             ):
#                 img_list.append(file_path)
#                 image_list.append(f)
#             else:
#                 all_filenames = [x[0] for x in os.walk(file_path)]
#                 for i in all_filenames:
#                     for f in glob.glob(os.path.join(i, "*.jpg")):
#                         img_list.append(f)
#     elif not os.path.splitext(dataset)[1] == "":
#         img_list = [dataset]
#         image_list = [dataset.split("/")[-1]]
#     else:
#         if dataset[-1] == "/":
#             dataset = dataset[:-1]
#             img_list = []
#             image_list = []
#             for f in os.listdir(dataset):
#                 file_path = os.path.join(dataset, f)
#                 img_list.append(file_path)
#                 image_list.append(f)
#     return img_list, image_list, filename, folder_name


# def save(folder_name):
#     all_filenames = [x[0] for x in os.walk(folder_name)]
#     all_filenames = all_filenames[1:]
#     full_path = []
#     folder_name = folder_name.split("/")
#     folder = folder_name[-1]
#     for f in all_filenames:
#         full_path.append(os.path.abspath(f))
#         classes_dir = []
#     for files in all_filenames:
#         classes_dir.append(files.split("/")[-1])
#     for classes in classes_dir:
#         os.makedirs(
#             os.path.abspath("Data_prep_images") + "/" + "dataset" + "/" + classes
#         )
#     for each_image in os.listdir("Data_prep_images/"):
#         # print(each_image)
#         path = os.path.abspath("Data_prep_images/" + each_image)
#         ls = os.path.splitext(path)[0]
#         ls = ls.split("/")
#         ls = ls[-1]
#         ls = ls.split("_")
#         rs = ls[0]
#         for i in classes_dir:
#             if i == rs:
#                 src = os.path.abspath(
#                     os.path.abspath("Data_prep_images/") + "/" + each_image
#                 )
#                 des = os.path.abspath(
#                     os.path.abspath("Data_prep_images/") + "/" + "dataset" + "/" + i
#                 )
#                 out = shutil.move(src, des)
#             else:
#                 out = None
#     # shutil.rmtree('Data_prep_images',ignore_errors=True)
#     shutil.make_archive("Data_prep_images", "zip", "Data_prep_images")
#     return out


#{"rotate":{"rotate_limit":90, "probability":1}, "blur":{"blur_limit_lower":9, "blur_limit_upper":17, "probability":1}}