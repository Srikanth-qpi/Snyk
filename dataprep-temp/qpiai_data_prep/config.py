import os
from typing import Optional

from pydantic import BaseModel

DB_LOG = os.getenv("DB_LOG", "true").lower() == "true"


# Use configure your input requirements in this model
class InputModel(BaseModel, extra="allow"):
    datatype: str = "tab"
    dataset: str = "link"
    zipfile:Optional[str] = "link"
    target_device: str = "cpu"
    num_device: int = 1
    data_prepcmd: str = "rbc"
    cmd_info: Optional[dict] = {}
    dataset_format: Optional[str] = 'coco'
    clmn: str = ""
    freq: Optional[int] = 1
    dataframe_delimiter: Optional[str] = ","
    category_column_name: Optional[str] = ""
    num_components: Optional[int] = 1
    perplexity: Optional[int] = 1
    all_column: Optional[bool] = False
    is_image: Optional[bool] = False
    inc_data: Optional[bool] = True
    shift_limit: Optional[int] = 0.0625
    scale_limit: Optional[int] = 0.1
    rotate_limit: Optional[int] = 90
    width: Optional[int] = 224
    height: Optional[int] = 224
    quality_lower: Optional[int] = 99
    quality_upper: Optional[int] = 100
    blur_limit_lower: Optional[int] = 3
    blur_limit_upper: Optional[int] = 7
    multi_min: Optional[int] = 0.9
    multi_max: Optional[int] = 1.1
    n_holes: Optional[int] = 8
    max_height: Optional[int] = 8
    max_width: Optional[int] = 8
    alpha_elastic: Optional[int] = 1
    alpha_affine: Optional[int] = 50
    sigma: Optional[int] = 50
    hue_shift_limit: Optional[int] = 20
    sat_shift_limit: Optional[int] = 30
    val_shift_limit: Optional[int] = 20
    distort_limit: Optional[int] = 0.3
    distort_limit_opt: Optional[int] = 0.05
    shift_limit_opt: Optional[int] = 0.05
    bright_limit: Optional[int] = 0.2
    contrast_limit: Optional[int] = 0.2
    r_shift: Optional[int] = 20
    g_shift: Optional[int] = 20
    b_shift: Optional[int] = 20
    alpha_fpca: Optional[int] = 0.1
    scale_min: Optional[int] = 0.25
    scale_max: Optional[int] = 0.25
    hue: Optional[int] = 0.2
    saturation: Optional[int] = 0.2
    channel_drop_max: Optional[int] = 1
    x_min: Optional[int] = None
    x_max: Optional[int] = None
    y_min: Optional[int] = None
    y_max: Optional[int] = None
    r_mean: Optional[int] = None
    g_mean: Optional[int] = None
    b_mean: Optional[int] = None
    r_std: Optional[int] = None
    b_std: Optional[int] = None
    g_std: Optional[int] = None
    probability: Optional[int] = 1


# RequestModel will include all the base metadata and things
class RequestModel(BaseModel):
    request_id: str = "507f1f77bcf86cd799439011"
    user_id: str = "test_user1"
    feature_type: str = "dataprep"
    input: InputModel = InputModel()


sample_input = {
    "request_id": "507f1f77bcf86cd799439011",
    "user_id": "test_user1",
    "feature_type": "qpiai-microservice-template",
    "input": {
        "datatype": "tab",
        "dataset": "link",
        "target_device": "cpu",
        "num_device": 1,
        "data_prepcmd": "rbc",
    },
}
