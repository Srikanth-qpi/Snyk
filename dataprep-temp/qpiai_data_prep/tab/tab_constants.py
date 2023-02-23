import json
import os

import pandas as pd
import qpiai as ap
from validator_collection import checkers

MULTI_DROP_DOWN = {"Name": "Select Columns", "Type": "MultiDropDown"}
SINGLE_DROP_DOWN = {"Name": "Select Single Column", "Type": "SingleDropDown"}
NUMBER = {"Name": "Select Number", "Type": "Number"}
STRING = {"Name": "Select String", "Type": "String"}
CHECKBOX = {"Name": "Select Checkbox", "Type": "CheckBox"}


def update_dict(pr_dict, upd_dict):
    pr_dict.update(upd_dict)
    return pr_dict


ARGUMENTS_DICT = {
    "train_test_split_classification": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select a Single Categorical column", "required": True},
        )
    },
    "train_test_split_regression": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select a Single Target column", "required": True},
        )
    },
    "isolation_forest_univariate": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(), {"Name": "Select Numeric Columns", "required": True}
        ),
        "all_column": update_dict(CHECKBOX.copy(), {"Name": "Select All Columns"}),
    },
    "1D scatter_plot": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(), {"Name": "Select Numeric Columns", "required": True}
        ),
        "category_column_name": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select Single Categorical Column", "required": True},
        ),
        "all_column": update_dict(
            CHECKBOX.copy(),
            {
                "Name": "Select All Columns",
                "Description": "Dataset should have numerical columns only",
            },
        ),
    },
    "2D scatter_plot": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(), {"Name": "Select Numeric Columns", "required": True}
        ),
        "category_column_name": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select Single Categorical Column", "required": True},
        ),
        "all_column": update_dict(
            CHECKBOX.copy(),
            {
                "Name": "Select All Columns",
                "Description": "Dataset should have numerical columns only",
            },
        ),
    },
    "3D scatter_plot": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(), {"Name": "Select Numeric Columns", "required": True}
        ),
        "category_column_name": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select Single Categorical Column", "required": True},
        ),
        "all_column": update_dict(
            CHECKBOX.copy(),
            {
                "Name": "Select All Columns",
                "Description": "Dataset should have numerical columns only",
            },
        ),
    },
    "bar_plot_missing_values": {
        "clmn": update_dict(MULTI_DROP_DOWN.copy(), {"required": True}),
        "all_column": update_dict(CHECKBOX.copy(), {"Name": "Select All Columns"}),
    },
    "line_plot": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(), {"Name": "Select Numeric Columns", "required": True}
        ),
        "category_column_name": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select Single Categorical Column", "required": True},
        ),
        "all_column": update_dict(
            CHECKBOX.copy(),
            {
                "Name": "Select All Columns",
                "Description": "Dataset should have numerical columns only",
            },
        ),
    },
    "histogram": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select Single Numerical Column", "required": True},
        ),
    },
    "PCA": {
        "category_column_name": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select Single Categorical Column", "required": True},
        ),
        "num_components": update_dict(
            NUMBER.copy(),
            {
                "Name": "Select Number of Components",
                "Description": "Please give Natural Number",
                "required": True,
            },
        ),
    },
    "tsne": {
        "category_column_name": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select Single Categorical Column", "required": True},
        ),
        "num_components": update_dict(
            NUMBER.copy(),
            {
                "Description": "Please give Natural Number",
                "Name": "Select Number of Components",
                "required": True,
            },
        ),
        "perplexity": update_dict(
            NUMBER.copy(),
            {
                "Description": "Please give Natural Number preferably between 5 and 50",
                "Name": "Select Perplexity",
                "required": False,
            },
        ),
    },
    "isolation_forest_bivariate": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(), {"Name": "Select Numeric Columns", "required": True}
        ),
        "all_column": update_dict(CHECKBOX.copy(), {"Name": "Select All Columns"}),
    },
    "isolation_forest_trivariate": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(), {"Name": "Select Numeric Column", "required": True}
        ),
        "all_column": update_dict(CHECKBOX.copy(), {"Name": "Select All Columns"}),
    },
    "outlier_interquartile": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select a numeric column", "required": True},
        )
    },
    "outlier_zscore": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select a numeric column", "required": True},
        )
    },
    "quantile_plot": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select a numeric column", "required": True},
        )
    },
    "LabelBinarizer": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Categorical Columns", "required": True},
        )
    },
    "Binarizer": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "FunctionTransformer": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "LabelEncoder": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Categorical Columns", "required": True},
        )
    },
    "MultiLabelBinarizer": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Categorical Columns", "required": True},
        )
    },
    "MaxAbsScaler": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "MinMaxScaler": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "StandardScaler": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "RobustScaler": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "Normalizer": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "OneHotEncoder": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Categorical Columns", "required": True},
        )
    },
    "OrdinalEncoder": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Categorical Columns", "required": True},
        )
    },
    "set_index": {"clmn": update_dict(SINGLE_DROP_DOWN.copy(), {"required": True})},
    "pearson_correlation": {},
    "spearman_correlation": {},
    "dist_plot": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {"Name": "Select One Numerical Column", "required": True},
        )
    },
    "count_plot": {"clmn": update_dict(SINGLE_DROP_DOWN.copy(), {"required": True})},
    "impute_mean": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "impute_median": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "impute_mode": {"clmn": update_dict(MULTI_DROP_DOWN.copy(), {"required": True})},
    "impute_drop": {},
    "impute_drop_subset": {
        "clmn": update_dict(MULTI_DROP_DOWN.copy(), {"required": True})
    },
    "impute_drop_column": {
        "clmn": update_dict(MULTI_DROP_DOWN.copy(), {"required": True})
    },
    "impute_interpolation": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "box_plot": {
        "clmn": update_dict(
            MULTI_DROP_DOWN.copy(),
            {"Name": "Select Numerical Columns", "required": True},
        )
    },
    "Summary": {},
    "variance_inflation_factor": {},
    "Impute_ExtraTreeRegressor": {},
    "Impute_KNN": {},
    "Impute_LinearRegression": {},
    "ft_imp_class": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {
                "Name": "Select a single column",
                "Description": "Please select the target column",
                "required": True,
            },
        )
    },
    "ft_imp_reg": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {
                "Name": "Select a single column",
                "Description": "Please select the target column",
                "required": True,
            },
        )
    },
    "chi_sq": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {
                "Name": "Select a single column",
                "Description": "Please select the target column",
                "required": True,
            },
        )
    },
    "f_classif": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {
                "Name": "Select a single column",
                "Description": "Please select the target column",
                "required": True,
            },
        )
    },
    "f_regress": {
        "clmn": update_dict(
            SINGLE_DROP_DOWN.copy(),
            {
                "Name": "Select a single column",
                "Description": "Please select the target column",
                "required": True,
            },
        )
    },
}


def get_uicomp_desc(prep_command, dataset=None, dataframe_delimiter=None):
    if isinstance(prep_command, list):
        prep_command_list = prep_command
    else:
        prep_command_list = [prep_command]

    data_resp = {}
    for pc in prep_command_list:
        ui_comps = ARGUMENTS_DICT[pc]

        if dataframe_delimiter == "Tab":
            dataframe_delimiter = "\t"
        if dataframe_delimiter == "Space":
            dataframe_delimiter = r"\s+"

        # print("For this ",pc,"dataset is ",dataset)
        print(ui_comps.items())
        for key, val in ui_comps.items():
            if val["Type"] in ["MultiDropDown", "SingleDropDown"]:
                if dataset is not None:
                    if checkers.is_url(dataset):
                        filename = ap.download(dataset)
                        if os.path.splitext(filename)[1] == ".zip":
                            # print("Filename is : ",filename)
                            file_csv = ap.unzip(filename)
                            dataset = file_csv
                            df = pd.read_csv(file_csv, sep=dataframe_delimiter, nrows=0)
                            columns = list(df)
                        else:
                            dataset = filename
                            df = pd.read_csv(filename, sep=dataframe_delimiter, nrows=0)
                            columns = list(df)
                    else:
                        # print("In else")
                        filename = dataset
                        if os.path.splitext(filename)[1] == ".zip":
                            # print("Filename is : ",filename)
                            file_csv = ap.unzip(filename)
                            dataset = file_csv
                            df = pd.read_csv(file_csv, sep=dataframe_delimiter, nrows=0)
                            columns = list(df)
                        else:
                            dataset = filename
                            df = pd.read_csv(filename, sep=dataframe_delimiter, nrows=0)
                            columns = list(df)

                    val.update({"Values": columns})
                else:
                    return {"status": False, "message": "Please provide a dataset"}

        data_resp[pc] = ui_comps

    return {"data": data_resp, "status": True}


"""
DATATYPE_DICT = {
    "impute_mean":{
        "clmn_type": "Numeric",
        "categorical_column_name_type"
    },
    "LabelEncoder":{
        "clmn_type": "Categorical"
    },
    "count_plot":{
        "clmn_type": "all"
    }
}

def get_filtered_columns(prep_command,val,df):
    if prep_command not in DATATYPE_DICT:
        return list(df.columns)

    else:
        for key,val in val.items():
            if val+"_type" in DATATYPE_DICT['prep_command'] == "Categorical":
                columns = fetch_cat_columns(df)
            if if val+"_type" in DATATYPE_DICT['prep_command'] == "Numeric":
                columns = fetch_numeric_columns(df)
        return columns
"""
