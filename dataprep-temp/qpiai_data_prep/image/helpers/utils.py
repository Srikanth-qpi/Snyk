import os

import albumentations as A
import cv2
import numpy as np


class Img_ops:

    export_dir = os.path.join(os.getcwd(), "Data_prep_images")

    def __init__(self):
        pass

    def edge_detection(image, name):
        median = cv2.medianBlur(image, 15)
        canny2 = cv2.Canny(median, 50, 150)
        return cv2.imwrite(os.path.join(Img_ops.export_dir, name), canny2)

    def enhancement(image, name):
        gamma = 1
        image_enhanced = np.power(image, gamma)
        return cv2.imwrite(os.path.join(Img_ops.export_dir, name), image_enhanced)

    def corner_detection(image, name):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_image, 100, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, 255, -1)
            result = cv2.imwrite(os.path.join(Img_ops.export_dir, name), image)
        return result

    def gray_scale(image, name):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.imwrite(os.path.join(Img_ops.export_dir, name), gray_image)

    def base_transform(data_prepcmd, image, d_format, name, **kwargs):
        
        cmd_info = kwargs.get('cmd_info', {})
        funcs = cmd_info.keys()



        args_transform = {
            "blur": A.Blur(
                blur_limit=(
                    kwargs.get("blur_limit_lower", 3),
                    kwargs.get("blur_limit_upper", 7),
                ),
                p=kwargs.get("probability", 0.5),
            ),
            "color_jitter": A.ColorJitter(
                brightness=kwargs.get("bright_limit", 0.2),
                contrast=kwargs.get("contrast_limit", 0.2),
                saturation=kwargs.get("saturation", 0.2),
                hue=kwargs.get("hue", 0.2),
                p=1,
            ),
            "clahe": A.CLAHE(p=1),
            "channel_shuffle": A.ChannelShuffle(p=1),
            "random_fog": A.RandomFog(p=1),
            "random_rain": A.RandomRain(p=1),
            "random_shadow": A.RandomShadow(p=1),
            "random_sunflare": A.RandomSunFlare(p=1),
            "random_snow": A.RandomSnow(p=1),
            "rbc": A.RandomBrightnessContrast(
                brightness_limit=kwargs.get("bright_limit", 0.2),
                contrast_limit=kwargs.get("contrast_limit", 0.2),
                p=1,
            ),
            "rgb_shift": A.RGBShift(
                r_shift_limit=kwargs.get("r_shift", 20),
                g_shift_limit=kwargs.get("g_shift", 20),
                b_shift_limit=kwargs.get("b_shift", 20),
                p=1,
            ),
            "hue_sat_value": A.HueSaturationValue(
                hue_shift_limit=kwargs.get("hue_shift_limit", 20),
                sat_shift_limit=kwargs.get("sat_shift_limit", 30),
                val_shift_limit=kwargs.get("val_shift_limit", 20),
                p=1,),
            "vertical_flip": A.VerticalFlip(p=kwargs.get("probability", 0.5)),
            "horizontal_flip": A.HorizontalFlip(p=kwargs.get("probability", 0.5)),
            "flip": A.Flip(p=kwargs.get("probability", 0.5)),
            "sharpen": A.IAASharpen(p=kwargs.get("probability", 0.5)),
            "channel_dropout": A.ChannelDropout(
                channel_drop_range=(1, kwargs.get("channel_drop_max", 1)),
                p=kwargs.get("probability", 0.5),
            ),
            "downscale": A.Downscale(
                scale_min=kwargs.get("scale_min", 0.25),
                scale_max=kwargs.get("scale_max", 0.25),
                p=kwargs.get("probability", 0.5),
            ),
            "sepia": A.ToSepia(p=kwargs.get("probability", 0.5)),
            "gaussian_blur": A.GaussianBlur(
                blur_limit=(
                    kwargs.get("blur_limit_lower", 3),
                    kwargs.get("blur_limit_upper", 7),
                ),
                p=kwargs.get("probability", 0.5),
            ),
            "equalize": A.Equalize(p=kwargs.get("probability", 0.5)),
            "median_blur": A.MedianBlur(
                blur_limit=(
                    kwargs.get("blur_limit_lower", 3),
                    kwargs.get("blur_limit_upper", 7),
                ),
                p=kwargs.get("probability", 0.5),
            ),
            "gauss_noise": A.GaussNoise(p=kwargs.get("probability", 0.5)),
            "glass_blur": A.GlassBlur(p=kwargs.get("probability", 0.5)),
            "emboss": A.IAAEmboss(p=kwargs.get("probability", 0.5)),
            "posterize": A.Posterize(p=kwargs.get("probability", 0.5)),
            "transpose": A.Transpose(p=kwargs.get("probability", 0.5)),
            "std_normalize": A.Normalize(p=1),
            "normalize": A.Normalize(
                mean=(
                    kwargs.get("r_mean", None),
                    kwargs.get("g_mean", None),
                    kwargs.get("b_mean", None),
                ),
                std=(
                    kwargs.get("r_std", None),
                    kwargs.get("g_std", None),
                    kwargs.get("b_std", None),
                ),
                p=1,
            ),
            "multi_noise": A.MultiplicativeNoise(
                multiplier=(kwargs.get("multi_min", 0.9), kwargs.get("multi_max", 1.1)),
                p=kwargs.get("probability", 0.5),
            ),
            "crop": A.Crop(
                x_min=kwargs.get("x_min", None),
                y_min=kwargs.get("y_min", None),
                x_max=kwargs.get("x_max", None),
                y_max=kwargs.get("y_max", None),
                p=1,
            ),
            "center_crop": A.CenterCrop(
                height=kwargs.get("height", None), width=kwargs.get("width", None), p=1
            ),
            "pad_if_needed": A.PadIfNeeded(
                min_height=kwargs.get("height", 224),
                min_width=kwargs.get("width", 224),
                p=1,
            ),
            "jpeg_compression": A.JpegCompression(
                quality_lower=kwargs.get("quality_lower", 99),
                quality_upper=kwargs.get("quality_upper", 100),
                p=kwargs.get("probability", 0.5),
            ),
            "motion_blur": A.MotionBlur(
                blur_limit=(
                    kwargs.get("blur_limit_lower", 3),
                    kwargs.get("blur_limit_upper", 7),
                ),
                p=kwargs.get("probability", 0.5),
            ),
            "iso_noise": A.ISONoise(p=kwargs.get("probability", 0.5)),
            "invert_img": A.InvertImg(p=kwargs.get("probability", 0.5)),
            "resize": A.Resize(
                width=kwargs.get("width", None), height=kwargs.get("height", None), p=1
            ),
            "rotate": A.Rotate(
                limit=kwargs.get("rotate_limit", 90), p=kwargs.get("probability", 0.5)
            ),
            "fancy_pca": A.FancyPCA(
                alpha=kwargs.get("alpha_fpca", 0.1), p=kwargs.get("probability", 0.5)
            ),
        }

        transforms = [args_transform[func] for func in funcs]
        if d_format == 'classification':
            transform = A.Compose(transforms)
        elif d_format == 'coco':
            transform = A.Compose(transforms, bbox_params = A.BboxParams(format = d_format, label_fields=['category_ids']))
        elif d_format == 'yolo':
            transform = A.Compose(transforms, bbox_params = A.BboxParams(format = d_format, label_fields=['category_ids']))
        elif d_format == 'coco_segmentation':
            transform = A.Compose(transforms)
        # transformed = transform(image=image, bboxes=boxes, category_ids=category_ids)
        # transformed_image = transformed["image"]
        # transformed_bboxes = transformed['bboxes']
        #return cv2.imwrite(i, transformed_image)
        return transform
