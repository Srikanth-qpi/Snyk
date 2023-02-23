import os

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, chi2, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import *

from qpiai_data_prep.file_log_writer import LogWriter
from qpiai_data_prep.plots_comb import *


class corr:
    def __init__(self):
        pass

    @classmethod
    def save(cls, df, file_name):
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
        return checkpoint

    def chi_sq(df, clmn):
        num_col = df._get_numeric_data().columns
        df_num = df[df.columns.intersection(num_col)]
        df_cat = df[df.columns.difference(num_col)]
        df_cat = df_cat.apply(LabelEncoder().fit_transform)
        df = pd.concat([df_cat, df_num], axis=1)
        y = df[[clmn]]
        X = df.iloc[:, df.columns != clmn]
        chi = chi2(X, y)
        df_3 = pd.DataFrame(chi)
        df_3 = pd.DataFrame(data=df_3.values, columns=X.columns)
        df_3 = (
            df_3.T.reset_index()
            .rename(columns={"index": "Feature", 0: "Chi_Score", 1: "p_value"})
            .sort_values(by="p_value")
        )
        df_3["Should_be_Accepted_or_Rejected"] = np.where(
            df_3["p_value"] > 0.05, "Should_be_rejected", "Should_be_accepted"
        )
        df = df_3
        file_name = "Chi_Square_Importance_HIGH_to_LOW.csv"
        return df, file_name

    def f_regress(df, clmn):
        num_col = df._get_numeric_data().columns
        df_num = df[df.columns.intersection(num_col)]
        df_cat = df[df.columns.difference(num_col)]
        df_cat = df_cat.apply(LabelEncoder().fit_transform)
        df = pd.concat([df_cat, df_num], axis=1)
        y = df[[clmn]]
        X = df.iloc[:, df.columns != clmn]
        freg = f_regression(X, y)
        df_3 = pd.DataFrame(freg)
        df_3 = pd.DataFrame(data=df_3.values, columns=X.columns)
        df_3 = (
            df_3.T.reset_index()
            .rename(columns={"index": "Feature", 0: "F_Score", 1: "p_value"})
            .sort_values(by="p_value")
        )
        df_3["Should_be_Accepted_or_Rejected"] = np.where(
            df_3["p_value"] > 0.05, "Should_be_rejected", "Should_be_accepted"
        )
        df = df_3
        file_name = "F_Value_Regression_Importance_HIGH_to_LOW.csv"
        return df, file_name

    def f_classif(df, clmn):
        num_col = df._get_numeric_data().columns
        df_num = df[df.columns.intersection(num_col)]
        df_cat = df[df.columns.difference(num_col)]
        df_cat = df_cat.apply(LabelEncoder().fit_transform)
        df = pd.concat([df_cat, df_num], axis=1)
        y = df[[clmn]]
        X = df.iloc[:, df.columns != clmn]
        fclass = f_classif(X, y)
        df_3 = pd.DataFrame(fclass)
        df_3 = pd.DataFrame(data=df_3.values, columns=X.columns)
        df_3 = (
            df_3.T.reset_index()
            .rename(columns={"index": "Feature", 0: "F_Score", 1: "p_value"})
            .sort_values(by="p_value")
        )
        df_3["Should_be_Accepted_or_Rejected"] = np.where(
            df_3["p_value"] > 0.05, "Should_be_rejected", "Should_be_accepted"
        )
        df = df_3
        file_name = "ANOVA_Categorical_Importance_HIGH_to_LOW.csv"
        return df, file_name

    def ft_imp_reg(df, clmn):
        y = df[[clmn]]
        X = df.iloc[:, df.columns != clmn]
        rfecv = RFECV(
            estimator=LinearRegression(),
            step=10,
            cv=4,
            scoring="neg_mean_squared_error",
        )
        rfecv.fit(X, y)
        features = [f for f, s in zip(X.columns, rfecv.support_) if s]
        df = df[df.columns.intersection(features)]
        file_name = "Feature_Importance_Regression.csv"
        return df, file_name

    def ft_imp_class(df, clmn):
        num_col = df._get_numeric_data().columns
        df_num = df[df.columns.intersection(num_col)]
        df_cat = df[df.columns.difference(num_col)]
        df_cat = df_cat.apply(LabelEncoder().fit_transform)
        df = pd.concat([df_cat, df_num], axis=1)
        y = df[[clmn]]
        X = df.iloc[:, df.columns != clmn]
        rfecv = RFECV(
            estimator=RandomForestClassifier(n_estimators=100, random_state=101),
            step=10,
            cv=StratifiedKFold(10),
            scoring="accuracy",
        )
        rfecv.fit(X, y)
        features = [f for f, s in zip(X.columns, rfecv.support_) if s]
        df = df[df.columns.intersection(features)]
        file_name = "Feature_Importance_Classification.csv"
        return df, file_name

    def spearman_correlation(df, custom_dashboard):
        Spearman_Correlation(df, custom_dashboard)
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        X = df.select_dtypes(include=numerics)
        X.fillna(X.mean())
        df = X.corr(method="spearman")
        df.insert(loc=0, column="Index", value=list(X.columns))
        file_name = "spearmanc.csv"
        return df, file_name

    def pearson_correlation(df, custom_dashboard):
        Pearson_Correlation(df, custom_dashboard)
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        X = df.select_dtypes(include=numerics)
        X.fillna(X.mean())
        df = X.corr(method="pearson")
        df.insert(loc=0, column="Index", value=list(X.columns))
        file_name = "pearsonc.csv"
        return df, file_name

    def ttsc(df, clmn):
        y = df[[clmn]]
        cols = list(df.columns)
        cols.remove(clmn)
        cols = cols + [clmn]
        df.drop([clmn], axis=1, inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, stratify=y
        )
        df1 = pd.concat([x_train, y_train], axis=1, ignore_index=True)
        df2 = pd.concat([x_test, y_test], axis=1, ignore_index=True)
        df1.columns = cols
        df2.columns = cols
        file_name1 = "train.csv"
        file_name2 = "test.csv"
        return df1, df2, file_name1, file_name2

    def ttsr(df, clmn):
        y = df[[clmn]]
        cols = list(df.columns)
        cols.remove(clmn)
        cols = cols + [clmn]
        df.drop([clmn], axis=1, inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        df1 = pd.concat([x_train, y_train], axis=1, ignore_index=True)
        df2 = pd.concat([x_test, y_test], axis=1, ignore_index=True)
        df1.columns = cols
        df2.columns = cols
        file_name1 = "train.csv"
        file_name2 = "test.csv"
        return df1, df2, file_name1, file_name2

    def PCA(df, custom_dashboard, **kwargs):
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        is_image = kwargs["is_image"]
        category_column_name = kwargs["category_column_name"]
        if category_column_name is not None:
            category_column = df[category_column_name]
            df = df.drop(category_column_name, axis=1)
        X = df.select_dtypes(include=numerics)
        X.fillna(X.mean())
        pca = decomposition.PCA()
        pca.n_components = kwargs["num_components"]
        pca_data = pca.fit_transform(X)
        columns = list()
        for i in range(kwargs["num_components"]):
            columns.append("Principal_Column_" + str(i + 1))
        if category_column_name is not None:
            pca_data = np.vstack((pca_data.T, category_column)).T
            columns.append(category_column_name)
        df = pd.DataFrame(data=pca_data, columns=columns)
        file_name = "PCA_" + str(kwargs["num_components"]) + ".csv"
        sc = scatter_plots(
            df,
            min(kwargs["num_components"], 2),
            ("Principal_Column_1", "Principal_Column_2")
            if kwargs["num_components"] > 1
            else ("Principal_Column_1",),
            ("1st Principal", "2nd Principal")
            if kwargs["num_components"] > 1
            else ("1st Principal",),
            category_column_name,
            custom_dashboard,
            is_image=is_image,
        )
        return df, file_name, sc

    def tsne(df, custom_dashboard, **kwargs):
        assert kwargs["num_components"] > 1, "Please enter value greater than 1"
        is_image = kwargs["is_image"]
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        category_column_name = kwargs["category_column_name"]
        if category_column_name is not None:
            category_column = kwargs["category_column_name"]
            l = df[category_column]
            df.drop(category_column_name, axis=1, inplace=True)
        X = df.select_dtypes(include=numerics)
        X.fillna(X.mean())
        model = TSNE(
            n_components=kwargs["num_components"],
            perplexity=kwargs["perplexity"],
            n_iter=2500,
        )
        tsne_data = model.fit_transform(X)
        columns = []
        for i in range(kwargs["num_components"]):
            columns.append("Dim_" + str(i + 1))
        if category_column_name is not None:
            tsne_data = np.vstack((tsne_data.T, l)).T
            columns.append(category_column_name)
        del df
        df = pd.DataFrame(data=tsne_data, columns=columns)
        file_name = "tsne" + str(kwargs["num_components"]) + ".csv"
        sc = scatter_plots(
            df,
            2,
            ("Dim_1", "Dim_2"),
            ("Dimension 1", "Dimension 2"),
            category_column_name,
            custom_dashboard,
            is_image=is_image,
        )
        return df, file_name, sc
