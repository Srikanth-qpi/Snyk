import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import *


class impute:
    def __init__(self):
        pass

    def impute_mean(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            x = df[df[clmn_str] != np.nan][clmn_str]
            df[clmn_str] = X.replace(np.nan, x.mean())
        file_name = "mean.csv"
        return df, file_name

    def impute_mode(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            x = df[df[clmn_str] != np.nan][clmn_str]
            df[clmn_str] = X.replace(np.nan, x.mode()[0])
        file_name = "mode.csv"
        return df, file_name

    def impute_median(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            x = df[df[clmn_str] != np.nan][clmn_str]
            df[clmn_str] = X.replace(np.nan, x.median())
        file_name = "median.csv"
        return df, file_name

    def impute_drop(df):
        df = df.dropna()
        file_name = "drop.csv"
        return df, file_name

    def impute_drop_subset(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            df = df.dropna(subset=[clmn_str])
        file_name = "drop_row.csv"
        return df, file_name

    def impute_drop_column(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            df = df.drop(columns=[clmn_str])
        file_name = "drop_column.csv"
        return df, file_name

    def impute_interpolation(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = X.interpolate(method="linear", limit_direction="both")
        file_name = "inter.csv"
        return df, file_name

    def summary(df):
        df = df.describe(include="all")
        df.insert(
            loc=0,
            column="Index",
            value=[
                "Count",
                "unique",
                "top",
                "frequency",
                "Mean",
                "Std",
                "Minimum",
                "25%",
                "50%",
                "75%",
                "max",
            ],
        )
        file_name = "summary.csv"
        return df, file_name

    def impute_extratreeregressor(df):
        encoder = OrdinalEncoder()
        imputer = IterativeImputer(ExtraTreesRegressor())
        cat = ["object", "bool"]
        data = df.select_dtypes(include=cat)

        def encode(data):
            nonulls = np.array(data.dropna())
            impute_reshape = nonulls.reshape(-1, 1)
            impute_ordinal = encoder.fit_transform(impute_reshape)
            data.loc[data.notnull()] = np.squeeze(impute_ordinal)
            return data

        for columns in list(data.columns):
            encode(df[columns])
        df = pd.DataFrame(np.round(imputer.fit_transform(df)), columns=df.columns)
        file_name = "iter_impute.csv"
        return df, file_name

    def impute_linear_regressor(df):
        encoder = OrdinalEncoder()
        imputer = IterativeImputer(LinearRegression())
        cat = ["object", "bool"]
        data = df.select_dtypes(include=cat)

        def encode(data):
            nonulls = np.array(data.dropna())
            impute_reshape = nonulls.reshape(-1, 1)
            impute_ordinal = encoder.fit_transform(impute_reshape)
            data.loc[data.notnull()] = np.squeeze(impute_ordinal)
            return data

        for columns in list(data.columns):
            encode(df[columns])
        df = pd.DataFrame(np.round(imputer.fit_transform(df)), columns=df.columns)
        file_name = "lr.csv"
        return df, file_name

    def impute_knn(df):
        encoder = OrdinalEncoder()
        imputer = KNNImputer(n_neighbors=3, weights="uniform")
        cat = ["object", "bool"]
        data = df.select_dtypes(include=cat)

        def encode(data):
            # retains only non-null values
            nonulls = np.array(data.dropna())
            # reshapes the data for encoding
            impute_reshape = nonulls.reshape(-1, 1)
            # encode date
            impute_ordinal = encoder.fit_transform(impute_reshape)
            # Assign back encoded values to non-null values
            data.loc[data.notnull()] = np.squeeze(impute_ordinal)
            return data

        # create a for loop to iterate through each column in the data
        for columns in list(data.columns):
            encode(df[columns])
        data = pd.DataFrame(imputer.fit_transform(df))
        data.columns = list(df.columns)
        data = data.round()
        df = data
        file_name = "knn_imputed.csv"
        return df, file_name

    def outlier_zscore(df, clmn):
        z = np.abs(stats.zscore(df[clmn]))
        df1 = df[(z < 3)]
        df2 = df[(z > 3)]
        file_name1 = "no_outlier_z.csv"
        file_name2 = "outlier_z.csv"
        return df1, df2, file_name1, file_name2

    def outlier_interquartile(df, clmn):
        q1 = df[clmn].quantile(0.25)
        q3 = df[clmn].quantile(0.75)
        iqr = q3 - q1
        df1 = df[~((df[clmn] < (q1 - 1.5 * iqr)) | (df[clmn] > (q3 + 1.5 * iqr)))]
        df2 = df[((df[clmn] < (q1 - 1.5 * iqr)) | (df[clmn] > (q3 + 1.5 * iqr)))]
        file_name1 = "no_outlier_int.csv"
        file_name2 = "outlier_int.csv"
        return df1, df2, file_name1, file_name2
