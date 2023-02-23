import os
import random

import albumentations as A
import cv2
import numpy as np


class Img_aug:
    export_dir = os.path.join(os.getcwd(), "Data_prep_images")

    def __init__(self):
        pass

    def base_func(data_prepcmd, image, name, **kwargs):

        args = {
            "random_scale": A.RandomScale(
                scale_limit=kwargs.get("scale_limit", 0.1), p=1
            ),
            "rbc": A.RandomBrightnessContrast(
                brightness_limit=kwargs.get("bright_limit", 0.2),
                contrast_limit=kwargs.get("contrast_limit", 0.2),
                p=1,
            ),
            "random_crop": A.RandomCrop(
                height=kwargs.get("height", None), width=kwargs.get("width", None), p=1
            ),
            "shift_scale_rotate": A.ShiftScaleRotate(
                shift_limit=kwargs.get("shift_limit", 0.0625),
                scale_limit=kwargs.get("scale_limit", 0.1),
                rotate_limit=kwargs.get("rotate_limit", 90),
                p=1,
            ),
            "hue_sat_value": A.HueSaturationValue(
                hue_shift_limit=kwargs.get("hue_shift_limit", 20),
                sat_shift_limit=kwargs.get("sat_shift_limit", 30),
                val_shift_limit=kwargs.get("val_shift_limit", 20),
                p=1,
            ),
            "color_jitter": A.ColorJitter(
                brightness=kwargs.get("bright_limit", 0.2),
                contrast=kwargs.get("contrast_limit", 0.2),
                saturation=kwargs.get("saturation", 0.2),
                hue=kwargs.get("hue", 0.2),
                p=1,
            ),
            "random_brightness": A.RandomBrightness(
                limit=kwargs.get("bright_limit", 0.2), p=1
            ),
            "random_contrast": A.RandomContrast(
                limit=kwargs.get("contrast_limit", 0.2), p=1
            ),
            "rgb_shift": A.RGBShift(
                r_shift_limit=kwargs.get("r_shift", 20),
                g_shift_limit=kwargs.get("g_shift", 20),
                b_shift_limit=kwargs.get("b_shift", 20),
                p=1,
            ),
            "optical_distortion": A.OpticalDistortion(
                distort_limit=kwargs.get("distort_limit_opt", 0.5),
                shift_limit=kwargs.get("shift_limit_opt", 0.05),
                p=1,
            ),
            "grid_distortion": A.GridDistortion(
                distort_limit=kwargs.get("distort_limit", 0.3), p=1
            ),
            "cutout": A.Cutout(
                num_holes=kwargs.get("n_holes", 8),
                max_h_size=kwargs.get("max_height", 8),
                max_w_size=kwargs.get("max_width", 8),
                p=1,
            ),
            "channel_shuffle": A.ChannelShuffle(p=1),
            "clahe": A.CLAHE(p=1),
            "elastic_transform": A.ElasticTransform(
                alpha=kwargs.get("alpha_elastic", 1),
                sigma=kwargs.get("sigma", 50),
                alpha_affine=kwargs.get("alpha_affine", 50),
                p=1,
            ),
            "random_fog": A.RandomFog(p=1),
            "random_rain": A.RandomRain(p=1),
            "random_shadow": A.RandomShadow(p=1),
            "random_sunflare": A.RandomSunFlare(p=1),
            "random_snow": A.RandomSnow(p=1),
        }

        inc_data = kwargs.get("inc_data")
        prob = kwargs.get("probability", 0.5)
        if inc_data == True:
            for i in range(5):
                randm = random.uniform(0, 1)
                if randm > 1 - prob:
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    transform = args[data_prepcmd]
                    transformed = transform(image=image)
                    transformed_image = transformed["image"]
                    filename = (
                        name.split(".")[0] + "_" + str(i) + "." + name.split(".")[1]
                    )
                    cv2.imwrite(
                        os.path.join(Img_aug.export_dir, filename), transformed_image
                    )
            filename = name.split(".")[0] + "_" + str(6) + "." + name.split(".")[1]
            cv2.imwrite(os.path.join(Img_aug.export_dir, filename), image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = args[data_prepcmd]
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            return cv2.imwrite(
                os.path.join(Img_aug.export_dir, name), transformed_image
            )
