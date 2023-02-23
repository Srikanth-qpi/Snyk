import json
import os

import pandas as pd
from validator_collection import checkers

NUMBER = {"Name": "Select Number", "Type": "Number"}
STRING = {"Name": "Select String", "Type": "String"}
CHECKBOX = {"Name": "Select Checkbox", "Type": "CheckBox"}


def update_dict(pr_dict, upd_dict):
    pr_dict.update(upd_dict)
    return pr_dict


ARGUMENTS_DICT = {
    "grayscale": {},
    "blur": {  # ODD
        "blur_limit_lower": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:3",
                "Name": "Odd Lower Blur limit",
                "required": False,
            },
        ),
        "blur_limit_upper": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:7",
                "Name": "Odd Upper Blur limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "corner_detection": {},
    "edge_detection": {},
    "enhancement": {},
    "vertical_flip": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "horizontal_flip": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "flip": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "sharpen": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "channel_shuffle": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "channel_dropout": {
        "channel_drop_max": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Integer(less than no. of channel), Default:1",
                "Name": "Max. channel to be dropped",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "clahe": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "color_jitter": {
        "bright_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Brightness",
                "required": False,
            },
        ),
        "contrast_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Contrast",
                "required": False,
            },
        ),
        "saturation": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Saturation",
                "required": False,
            },
        ),
        "hue": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Hue",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "downscale": {
        "scale_min": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input(Should be < 1), Default:0.25",
                "Name": "Minimum Scale",
                "required": False,
            },
        ),
        "scale_max": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input(Should be < 1), Default:0.25",
                "Name": "Maximum Scale",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "sepia": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "fancy_pca": {
        "alpha_fpca": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.1",
                "Name": "Alpha",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "gaussian_blur": {  # ODD
        "blur_limit_lower": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:3",
                "Name": "Odd Lower Blur limit",
                "required": False,
            },
        ),
        "blur_limit_upper": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:7",
                "Name": "Odd Upper Blur limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "equalize": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "rgb_shift": {
        "r_shift": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:20",
                "Name": "R Shift Limit",
                "required": False,
            },
        ),
        "g_shift": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:20",
                "Name": "G Shift Limit",
                "required": False,
            },
        ),
        "b_shift": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:20",
                "Name": "B Shift Limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "random_brightness": {
        "bright_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Brightness",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "random_contrast": {
        "contrast_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Contrast",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "rbc": {
        "bright_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Brightness",
                "required": False,
            },
        ),
        "contrast_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.2",
                "Name": "Contrast",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "median_blur": {  # ODD
        "blur_limit_lower": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:3",
                "Name": "Odd Lower Blur limit",
                "required": False,
            },
        ),
        "blur_limit_upper": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:7",
                "Name": "Odd Upper Blur limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "optical_distortion": {
        "distort_limit_opt": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.05",
                "Name": "Distortion Limit",
                "required": False,
            },
        ),
        "shift_limit_opt": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.05",
                "Name": "Shift Limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "gauss_noise": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "glass_blur": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "emboss": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "grid_distortion": {
        "distort_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.3",
                "Name": "Distortion Limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "hue_sat_value": {
        "hue_shift_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:20",
                "Name": "Hue Shift Limit",
                "required": False,
            },
        ),
        "sat_shift_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:30",
                "Name": "Saturation Shift Limit",
                "required": False,
            },
        ),
        "val_shift_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:20",
                "Name": "Value Shift Limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "elastic_transform": {
        "alpha_elastic": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:1",
                "Name": "Alpha",
                "required": False,
            },
        ),
        "alpha_affine": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:50",
                "Name": "Alpha Affine",
                "required": False,
            },
        ),
        "sigma": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:50",
                "Name": "Sigma",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "posterize": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "random_fog": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "random_rain": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "random_shadow": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "random_sunflare": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "random_snow": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "cutout": {
        "n_holes": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:8",
                "Name": "Number of holes",
                "required": False,
            },
        ),
        "max_height": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:8",
                "Name": "Maximum Height",
                "required": False,
            },
        ),
        "max_width": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:8",
                "Name": "Maximum Width",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "transpose": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "std_normalize": {},
    "normalize": {
        "r_mean": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "R Mean",
                "required": True,
            },
        ),
        "g_mean": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "G Mean",
                "required": True,
            },
        ),
        "b_mean": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "B Mean",
                "required": True,
            },
        ),
        "r_std": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "R Std Deviation",
                "required": True,
            },
        ),
        "g_std": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "G Std Deviation",
                "required": True,
            },
        ),
        "b_std": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "B Std Deviation",
                "required": True,
            },
        ),
    },
    "multi_noise": {
        "multi_min": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.9",
                "Name": "Lower Multiplier",
                "required": False,
            },
        ),
        "multi_max": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:1.1",
                "Name": "Upper Multiplier",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "random_crop": {
        "width": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Width",
                "required": True,
            },
        ),
        "height": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Height",
                "required": True,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "crop": {
        "x_min": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "X Min.",
                "required": True,
            },
        ),
        "x_max": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "X Max.",
                "required": True,
            },
        ),
        "y_min": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Y Min.",
                "required": True,
            },
        ),
        "y_max": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Y Max.",
                "required": True,
            },
        ),
    },
    "center_crop": {
        "width": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Width",
                "required": True,
            },
        ),
        "height": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Height",
                "required": True,
            },
        ),
    },
    "pad_if_needed": {
        "width": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:224",
                "Name": "Width",
                "required": True,
            },
        ),
        "height": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:224",
                "Name": "Height",
                "required": True,
            },
        ),
    },
    "motion_blur": {  # ODD
        "blur_limit_lower": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:3",
                "Name": "Odd Lower Blur limit",
                "required": False,
            },
        ),
        "blur_limit_upper": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Odd Input(3 or more), Default:7",
                "Name": "Odd Upper Blur limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "jpeg_compression": {
        "quality_lower": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:99",
                "Name": "Lower Quality",
                "required": False,
            },
        ),
        "quality_upper": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:100",
                "Name": "Upper Quality",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "iso_noise": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "invert_img": {
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        )
    },
    "resize": {
        "width": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Width",
                "required": True,
            },
        ),
        "height": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:None",
                "Name": "Height",
                "required": True,
            },
        ),
    },
    "random_scale": {
        "scale_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.1",
                "Name": "Scale Limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
    "rotate": {
        "rotate_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:90",
                "Name": "Rotate Limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
    },
    "shift_scale_rotate": {
        "shift_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.0625",
                "Name": "Shift Limit",
                "required": False,
            },
        ),
        "rotate_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:90",
                "Name": "Rotate Limit",
                "required": False,
            },
        ),
        "scale_limit": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input, Default:0.1",
                "Name": "Scale Limit",
                "required": False,
            },
        ),
        "probability": update_dict(
            NUMBER.copy(),
            {
                "Description": "Provide Input between 0 & 1, Default:0.5",
                "Name": "Probability",
                "required": False,
            },
        ),
        "inc_data": update_dict(CHECKBOX.copy(), {"Name": "Don't increase Data"}),
    },
}


def get_uicomp_desc(prep_command, **kwargs):
    if isinstance(prep_command, list):
        prep_command_list = prep_command
    else:
        prep_command_list = [prep_command]

    data_resp = {}
    for pc in prep_command_list:
        ui_comps = ARGUMENTS_DICT[pc]
        print(ui_comps.items())
        # for key,val in ui_comps.items():
        data_resp[pc] = ui_comps

    return {"data": data_resp, "status": True}
    # return {prep_command:ui_comps,"status":True}

    # curl -d '{"data_prepcmd":"rgb_shift","datatype":"image"}' -H 'Content-Type: application/json' localhost:5002/qpiai_dataprepuicomp
