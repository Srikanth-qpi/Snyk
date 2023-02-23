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
#from temp.db_info import db

def image_dataprep(dataset, dataset_format, target_device, num_device, data_prepcmd=None, **kwargs):

    data = Dataset(dataset, dataset_format)
    img_list, image_list, annot_list, filename, folder_name = data.download()
    print(len(img_list))
    print(filename, folder_name)

    cmd_info = kwargs.get('cmd_info', {})


    if dataset_format == 'coco':
        with open(annot_list[0], 'r') as f:
            coco = json.load(f)
        images = coco['images']
        annotations = coco['annotations']
        count_images = annotations[-1]['image_id'] + 1
        image_count = images[-1]['id'] + 1
        count = 0
        print('len', len(img_list))
        for i,k  in zip(images, img_list):
            count = count + 1
            print(count)
            image = cv2.imread(k)
            image_id = i['id']
            image_name = i['file_name']
            root_image_path = os.path.dirname(img_list[0])
            #print(root_image_path)
            image_path = os.path.join(root_image_path, image_name)
            image = cv2.imread(image_path)
            boxes = []
            category_ids = []
            for j in annotations:
                if j["image_id"] == image_id:
                    boxes.append(j['bbox'])
                    category_ids.append(j['category_id'])
            print(boxes, category_ids)
            #transformed = transform(image=image, bboxes=boxes, category_ids=category_ids)
            transform = Img_ops.base_transform(data_prepcmd=None, image=image, name=k, **kwargs)
            #print(transform)
            transformed = transform(image=image, bboxes=boxes, category_ids=category_ids)
            transformed_image = transformed["image"]
            transformed_bboxes = transformed['bboxes']
            #print(transformed_bboxes)
            #exit()
            count_id = annotations[-1]['id'] + 1
            for tb, c in zip(transformed_bboxes, category_ids):
                entry_annotations = {'id': count_id,
                        'image_id': count_images,
                        'category_id': c,
                        'bbox': [int(c) for c in tb],
                        'area': 0,
                        'segmentation': [],
                        'iscrowd': 0,
                        "attributes": {"occluded": False, "rotation": 0.0}
                        }
                annotations.append(entry_annotations)
                count_id = count_id + 1
            count_images = count_images + 1
            transformed_image = transformed['image']
            entry_images = {'id': image_count,
            'license': 1,
            'file_name': str(os.path.splitext(image_name)[0]) + '_mod' + str(os.path.splitext(image_name)[1]),
            'height': transformed_image.shape[0],
            'width': transformed_image.shape[1],
            'date_captured': '2020-09-20T23:43:09+00:00'}
            images.append(entry_images)
            image_count = image_count + 1
            path = root_image_path + '/' + str(os.path.splitext(image_name)[0]) + '_mod' + str(os.path.splitext(image_name)[1])
            cv2.imwrite(path,transformed_image)
        #print(annotations)
        with open(annot_list[0], 'w') as f:
            json.dump(coco, f)
        shutil.make_archive(folder_name, "zip", folder_name)

    elif dataset_format == 'yolo':
        for i,j in zip(img_list, annot_list):
            mod_imagename = os.path.dirname(i) + '/' + i.split('/')[-1].split('.')[0]   + '_mod' + '.' + i.split('/')[-1].split('.')[-1]
            print('mod_imagename',mod_imagename)
            print('filename',j)
            mod_filename = os.path.dirname(j) + '/' + j.split('/')[-1].split('.')[0] + '_mod' + '.txt'
            print('mod_filename',mod_filename)
            image = cv2.imread(i)
            boxes = list()
            category_ids = list()
            with open(j) as f:
                lines = f.readlines()
                print('lines', lines)
                for i in range(len(lines)):
                    r = lines[i].strip()
                    l = r.split(' ')
                    l = [float(x) for x in l]
                    category_ids.append(int(l[0]))
                    #a_list = [l[i] for i in order]
                    #a_list[3] = int(a_list[3])
                    #print(a_list)
                    #boxes.append(a_list)
                    boxes.append(l[1:])
            print('og_boxes', boxes)
            transform = Img_ops.base_transform(data_prepcmd=None, image=image, d_format=dataset_format,  name=i, **kwargs)
            transformed = transform(image=image, bboxes=boxes, category_ids=category_ids)
            transformed_image = transformed["image"]
            cv2.imwrite(mod_imagename, transformed_image)
            transformed_bboxes = transformed['bboxes']
            print('bboxes', transformed_bboxes)
            values = list()
            for i, j in zip(transformed_bboxes, category_ids):
                delimiter = ','
                c = delimiter.join([str(value) for value in i])
                c = c.replace(',', ' ')
                d = str(j) + ' ' + c
                values.append(d)
            f = open(mod_filename, 'a+')
            lines = f.readlines()
            f.truncate(0)
            for value in values:
                f.write(value)
                f.write('\n')
            f.close()
        shutil.make_archive(folder_name, "zip", folder_name)

    elif dataset_format == 'classification':
        for img in img_list:
            print('len', len(img_list))
            print('path', img)
            mod_imagename = os.path.dirname(img) + '/' + img.split('/')[-1].split('.')[0]   + '_mod' + '.' + img.split('/')[-1].split('.')[-1]
            print('mod', mod_imagename)
            name = Path(img).name
            image = cv2.imread(img)
            transform = Img_ops.base_transform(data_prepcmd=None, image=image, d_format=dataset_format,  name=img, **kwargs)
            transformed = transform(image=image)
            transformed_image = transformed['image']
            cv2.imwrite(mod_imagename, transformed_image)   
        shutil.make_archive(folder_name, "zip", folder_name)
      
    

    
    output_folder = folder_name + '.zip'
    print(os.path.abspath(output_folder))
    return {"dataPrepOutput": os.path.abspath(output_folder)}










    

    

if __name__ == "__main__":
    #image_dataprep(dataset= '/home/ec2-user/dataprep/task_test-2022_11_16_13_48_35-yolo 1.1.zip', dataset_format='yolo', target_device='cpu', num_device=1, cmd_info={"rotate":{"rotate_limit":90, "probability":1}, "blur":{"blur_limit_lower":9, "blur_limit_upper":17, "probability":1}})
    image_dataprep(dataset = 'https://qpiaidataset.s3.amazonaws.com/tiny_shoppee_train.zip', dataset_format = 'classification', target_device='cpu', num_device=1, cmd_info={"rotate":{"rotate_limit":90, "probability":1}, "blur":{"blur_limit_lower":9, "blur_limit_upper":17, "probability":1}})