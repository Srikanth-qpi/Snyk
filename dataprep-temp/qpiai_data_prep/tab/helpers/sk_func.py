import numpy as np
import pandas as pd
from sklearn.preprocessing import *


class sk_func:
    def __init__(self):
        pass

    def Robust_Scaler(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = RobustScaler().fit_transform(X)
        file_name = "Robscaler.csv"
        return df, file_name

    def Standard_Scaler(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = StandardScaler().fit_transform(X)
        file_name = "Stdscaler.csv"
        return df, file_name

    def MinMaxScaler(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = MinMaxScaler().fit_transform(X)
        file_name = "Minmax.csv"
        return df, file_name

    def MaxAbsScaler(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = MaxAbsScaler().fit_transform(X)
        file_name = "Maxabs.csv"
        return df, file_name

    def Label_Encoder(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            df[clmn_str] = LabelEncoder().fit_transform(df[clmn_str])
        file_name = "labelEn.csv"
        return df, file_name

    def Function_transformer(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = FunctionTransformer(np.sqrt).transform(X)
        file_name = "funcTrans.csv"
        return df, file_name

    def Binarizer(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = Binarizer().fit_transform(X)
        file_name = "Binarizer.csv"
        return df, file_name

    def Label_Binarizer(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            df[clmn_str] = LabelBinarizer().fit_transform(df[clmn_str])
        file_name = "LabelBin.csv"
        return df, file_name

    def Normalizer(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = Normalizer().fit_transform(X)
        file_name = "Normalizer.csv"
        return df, file_name

    def OneHot_Encoder(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            df = pd.get_dummies(df, columns=[clmn_str], prefix=["Type_is"])
        file_name = "OneHot.csv"
        return df, file_name

    def Ordinal_Encoder(df, clmn, **kwargs):
        column_names = clmn.split(kwargs["dataframe_delimiter"])
        for clmn_str in column_names:
            clmn_index = df.columns.get_loc(clmn_str)
            X = df.iloc[:, [clmn_index]]
            df[clmn_str] = OrdinalEncoder().fit_transform(X)
        file_name = "Normalizer.csv"
        return df, file_name
