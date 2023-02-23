import os

import albumentations as A
import cv2
import numpy as np


class Vid_ops:
    def __init__(self):
        pass

    def edge_detection(frame, out):
        median = cv2.medianBlur(frame, 15)
        canny2 = cv2.Canny(median, 50, 150)
        output = cv2.cvtColor(canny2, cv2.COLOR_GRAY2BGR)
        return out.write(output)

    def background_subtraction(frame, out):
        fgbg = cv2.createBackgroundSubtractorMOG2()
        fgmask = fgbg.apply(frame)
        return out.write(fgmask)

    def enhancement(frame, out):
        gamma = 0.4
        frame_enhanced = np.power(frame, gamma)
        return out.write(np.uint8(frame_enhanced))

    def blur(frame, out):
        median = cv2.medianBlur(frame, 15)
        return out.write(median)

    def corner_detection(frame, out):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_frame, 100, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, 255, -1)
            result = out.write(frame)
        return result

    def gray_scale(frame, out):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return out.write(gray_frame)

    def rbc(frame, out):
        transform = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def shift_scale_rotate(frame, out):
        transform = A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=90
        )
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def color_jitter(frame, out):
        transform = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def rgb_shift(frame, out):
        transform = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20)
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def channel_shuffle(frame, out):
        transform = A.ChannelShuffle()
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def random_fog(frame, out):
        transform = A.RandomFog()
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def random_rain(frame, out):
        transform = A.RandomRain()
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def random_shadow(frame, out):
        transform = A.RandomShadow()
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def random_sunflare(frame, out):
        transform = A.RandomSunFlare()
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)

    def random_snow(frame, out):
        transform = A.RandomSnow()
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        return out.write(transformed_image)
