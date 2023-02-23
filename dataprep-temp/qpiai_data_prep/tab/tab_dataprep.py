import io
import json
import os
import sys
from contextlib import redirect_stdout
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import decomposition
from sklearn.ensemble import (
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import RFECV, chi2, f_classif, f_regression
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from qpiai_data_prep.file_log_writer import LogWriter
from qpiai_data_prep.plots_comb import *
from qpiai_data_prep.tab.helpers.corr import *
from qpiai_data_prep.tab.helpers.data import *
from qpiai_data_prep.tab.helpers.graphs import *
from qpiai_data_prep.tab.helpers.impute import *
from qpiai_data_prep.tab.helpers.sk_func import *
from temp.db_info import db

def tab_dataprep(dataset, target_device, num_device, data_prepcmd, clmn, pipeline=False, **kwargs):
    if not pipeline:
        db.update_progress(progress=5)
    df = data.download(dataset, **kwargs)
    if not pipeline:
        db.update_progress(progress=25)

    df.head()
    value = df.columns.tolist()

    is_image = kwargs["is_image"]
    custom_dashboard = LogWriter(microservice="dataprep")
    if not pipeline:
        db.update_progress(progress=40)
    if data_prepcmd[3:] == "scatter_plot":
        sc = graphs.scatter_plot(df, clmn, custom_dashboard, data_prepcmd, **kwargs)

    elif data_prepcmd == "bar_plot_missing_values":
        bp = graphs.bar_plot(df, clmn, custom_dashboard, **kwargs)

    elif data_prepcmd == "line_plot":
        lp = graphs.line_plot(df, clmn, custom_dashboard, **kwargs)

    elif data_prepcmd == "quantile_plot":
        quantile_plots(df, clmn, custom_dashboard, is_image=is_image)

    elif data_prepcmd == "count_plot":
        countplot(df, clmn, custom_dashboard, is_image=is_image)

    elif data_prepcmd == "box_plot":
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        box_plot(df, column_names, custom_dashboard, is_image=is_image)

    elif data_prepcmd == "dist_plot":
        distplot(df, clmn, custom_dashboard, is_image=is_image)

    elif data_prepcmd == "histogram":
        histogram(df, clmn, custom_dashboard, is_image=is_image)

    elif data_prepcmd == "spearman_correlation":
        df, file_name = corr.spearman_correlation(df, custom_dashboard)

    elif data_prepcmd == "pearson_correlation":
        df, file_name = corr.pearson_correlation(df, custom_dashboard)

    elif data_prepcmd == "PCA":
        df, file_name, sc = corr.PCA(df, custom_dashboard, **kwargs)

    elif data_prepcmd == "tsne":
        df, file_name, sc = corr.tsne(df, custom_dashboard, **kwargs)

    elif data_prepcmd == "LabelBinarizer":
        df, file_name = sk_func.Label_Binarizer(df, clmn, **kwargs)

    elif data_prepcmd == "Binarizer":
        df, file_name = sk_func.Binarizer(df, clmn, **kwargs)

    elif data_prepcmd == "FunctionTransformer":
        df, file_name = sk_func.Function_transformer(df, clmn, **kwargs)

    elif data_prepcmd == "LabelEncoder":
        df, file_name = sk_func.Label_Encoder(df, clmn, **kwargs)

    elif data_prepcmd == "MaxAbsScaler":
        df, file_name = sk_func.MaxAbsScaler(df, clmn, **kwargs)

    elif data_prepcmd == "MinMaxScaler":
        df, file_name = sk_func.MinMaxScaler(df, clmn, **kwargs)

    elif data_prepcmd == "StandardScaler":
        df, file_name = sk_func.Standard_Scaler(df, clmn, **kwargs)

    elif data_prepcmd == "RobustScaler":
        df, file_name = sk_func.Robust_Scaler(df, clmn, **kwargs)

    elif data_prepcmd == "Normalizer":
        df, file_name = sk_func.Normalizer(df, clmn, **kwargs)

    elif data_prepcmd == "OneHotEncoder":
        df, file_name = sk_func.OneHot_Encoder(df, clmn, **kwargs)

    elif data_prepcmd == "OrdinalEncoder":
        df, file_name = sk_func.Ordinal_Encoder(df, clmn, **kwargs)

    elif data_prepcmd == "outlier_interquartile":
        df1, df2, file_name1, file_name2 = impute.outlier_interquartile(df, clmn)

    elif data_prepcmd == "outlier_zscore":
        df1, df2, file_name1, file_name2 = impute.outlier_zscore(df, clmn)

    elif (
        data_prepcmd.split("_")[-1] == "univariate"
        or data_prepcmd.split("_")[-1] == "bivariate"
        or data_prepcmd.split("_")[-1] == "trivariate"
    ):
        df1 = df.copy()
        df2 = df.copy()
        col_list = clmn.split(kwargs["dataframe_delimiter"])
        outlierPreds = np.ones((len(col_list), len(df)))
        for i, column in enumerate(col_list):
            isolation_forest = IsolationForest(contamination="auto")
            isolation_forest.fit(df[column].values.reshape(-1, 1))

            xx = np.linspace(df[column].min(), df[column].max(), len(df)).reshape(-1, 1)
            anomaly_score = isolation_forest.decision_function(xx)
            outlier = isolation_forest.predict(xx)

            outlierPreds[i] = isolation_forest.predict(df[column].values.reshape(-1, 1))
        drop_list = []
        tempArr = outlierPreds.T
        for i in range(0, len(tempArr)):
            count = 0
            for j in tempArr[i]:
                if j == -1:
                    count = count + 1
            if (
                (data_prepcmd.split("_")[-1] == "univariate" and count >= 1)
                or (data_prepcmd.split("_")[-1] == "bivariate" and count >= 2)
                or (data_prepcmd.split("_")[-1] == "trivariate" and count >= 3)
            ):
                df1.drop([i], axis=0, inplace=True)
                drop_list.append(i)
        df2 = df[df.index.isin(drop_list)]
        file_name1 = "no_outlier_forest.csv"
        file_name2 = "outlier_forest.csv"

    elif data_prepcmd == "train_test_split_classification":
        df1, df2, file_name1, file_name2 = corr.ttsc(df, clmn)

    elif data_prepcmd == "train_test_split_regression":
        df1, df2, file_name1, file_name2 = corr.ttsr(df, clmn)

    elif data_prepcmd == "ft_imp_class":
        df, file_name = corr.ft_imp_class(df, clmn)

    elif data_prepcmd == "ft_imp_reg":
        df, file_name = corr.ft_imp_reg(df, clmn)

    elif data_prepcmd == "impute_mean":
        df, file_name = impute.impute_mean(df, clmn, **kwargs)

    elif data_prepcmd == "impute_mode":
        df, file_name = impute.impute_mode(df, clmn, **kwargs)

    elif data_prepcmd == "impute_median":
        df, file_name = impute.impute_median(df, clmn, **kwargs)

    elif data_prepcmd == "impute_drop":
        df, file_name = impute.impute_drop(df)

    elif data_prepcmd == "impute_drop_subset":
        df, file_name = impute.impute_drop_subset(df, clmn, **kwargs)

    elif data_prepcmd == "impute_drop_column":
        df, file_name = impute.impute_drop_column(df, clmn, **kwargs)

    elif data_prepcmd == "impute_interpolation":
        df, file_name = impute.impute_interpolation(df, clmn, **kwargs)

    elif data_prepcmd == "Summary":
        df, file_name = impute.summary(df)

    elif data_prepcmd == "Impute_ExtraTreeRegressor":
        df, file_name = impute.impute_extratreeregressor(df)

    elif data_prepcmd == "Impute_LinearRegression":
        df, file_name = impute.impute_linear_regressor(df)

    elif data_prepcmd == "Impute_KNN":
        df, file_name = impute.impute_knn(df)

    elif data_prepcmd == "chi_sq":
        df, file_name = corr.chi_sq(df, clmn)

    elif data_prepcmd == "f_classif":
        df, file_name = corr.f_classif(df, clmn)

    elif data_prepcmd == "f_regress":
        df, file_name = corr.f_regress(df, clmn)

    if data_prepcmd in [
        "1D scatter_plot",
        "2D scatter_plot",
        "3D scatter_plot",
        "quantile_plot",
        "count_plot",
        "dist_plot",
        "box_plot",
        "histogram",
        "line_plot",
        "bar_plot_missing_values",
    ]:
        checkpoint = {
            "logsOutput": os.path.join(
                os.environ.get("LOGS_PATH_DIR", "."),
                "logs_" + os.environ.get("REQUEST_ID", "null"),
            )
        }

    elif data_prepcmd in ["spearman_correlation", "pearson_correlation", "PCA", "tsne"]:
        checkpoint = {
            "logsOutput": os.path.join(
                os.environ.get("LOGS_PATH_DIR", "."),
                "logs_" + os.environ.get("REQUEST_ID", "null"),
            )
        }
        df.to_csv(file_name, index=False)
        PATH = list()
        PATH.append(os.path.abspath(file_name))
        checkpoint.update({"dataPrepOutput": PATH})

    elif (
        data_prepcmd
        in [
            "outlier_interquartile",
            "outlier_zscore",
            "train_test_split_classification",
            "train_test_split_regression",
        ]
        or data_prepcmd.split("_")[-1] == "univariate"
        or data_prepcmd.split("_")[-1] == "bivariate"
        or data_prepcmd.split("_")[-1] == "trivariate"
    ):
        df1.to_csv(file_name1, index=False)
        df2.to_csv(file_name2, index=False)
        PATH = []
        PATH.append(os.path.abspath(file_name1))
        PATH.append(os.path.abspath(file_name2))
        checkpoint = dict({"dataPrepOutput": PATH})
    else:
        df.to_csv(file_name, index=False)
        PATH = list()
        PATH.append(os.path.abspath(file_name))
        checkpoint = dict({"dataPrepOutput": PATH})
    # checkpoint = os.path.abspath(filename)
    if not pipeline:
        db.update_progress(progress=95)
    return checkpoint
