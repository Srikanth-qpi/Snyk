import os
from itertools import combinations

import numpy as np
import pandas as pd

from qpiai_data_prep.file_log_writer import LogWriter
from qpiai_data_prep.plots_comb import *


class graphs:
    def __init__(self):
        pass

    def bar_plot(df, clmn, custom_dashboard, **kwargs):
        is_image = kwargs["is_image"]
        all_column = kwargs["all_column"]
        if clmn == "":
            column_names = list()
        else:
            column_names = clmn.split(kwargs["dataframe_delimiter"])
        column_names = df.columns.tolist() if all_column else column_names
        assert len(column_names) > 0, "No columns selected"
        bp = barplot(
            df, column_names, custom_dashboard, "missing_values", is_image=is_image
        )
        return bp

    def scatter_plot(df, clmn, custom_dashboard, data_prepcmd, **kwargs):
        is_image = kwargs["is_image"]
        category_column_name = kwargs["category_column_name"]
        dimensionality = int(data_prepcmd[0])
        if clmn == "":
            column_names = list()
        else:
            column_names = clmn.split(kwargs["dataframe_delimiter"])
        all_column = kwargs["all_column"]
        value = df.columns.tolist()
        if category_column_name is not None:
            value.remove(category_column_name)
        if dimensionality == 1:
            assert (
                len(column_names) + (1 if all_column else 0) == 1
            ), "Incorrect No. of columns provided"
            if all_column:
                for i in value:
                    sc = scatter_plots(
                        df,
                        dimensionality,
                        (i,),
                        (i,),
                        category_column_name,
                        custom_dashboard,
                        is_image=is_image,
                    )
            else:
                sc = scatter_plots(
                    df,
                    dimensionality,
                    column_names,
                    column_names,
                    category_column_name,
                    custom_dashboard,
                    is_image=is_image,
                )
        elif dimensionality == 2:
            assert (
                len(column_names) + (1 if all_column else 0)
            ) <= 2, "Incorrect No. of columns provided"
            assert (
                len(column_names) == 2 or all_column
            ), "Incorrect No. of columns provided"
            if not all_column:
                sc = scatter_plots(
                    df,
                    dimensionality,
                    column_names,
                    column_names,
                    category_column_name,
                    custom_dashboard,
                    is_image=is_image,
                )
            elif len(column_names) == 1:
                for i in value:
                    if i not in column_names:
                        sc = scatter_plots(
                            df,
                            dimensionality,
                            (i, column_names[0]),
                            (i, column_names[0]),
                            category_column_name,
                            custom_dashboard,
                            is_image=is_image,
                        )
            else:
                comb = combinations(value, dimensionality)
                for i in list(comb):
                    sc = scatter_plots(
                        df,
                        dimensionality,
                        i,
                        i,
                        category_column_name,
                        custom_dashboard,
                        is_image=is_image,
                    )
        elif dimensionality == 3:
            assert (
                len(column_names) + (1 if all_column else 0)
            ) <= 3, "Incorrect No. of columns provided"
            assert (
                len(column_names) == 3 or all_column
            ), "Incorrect No. of columns provided"
            if not all_column:
                sc = scatter_plots(
                    df,
                    dimensionality,
                    column_names,
                    column_names,
                    category_column_name,
                    custom_dashboard,
                    is_image=is_image,
                )
            elif len(column_names) == 2:
                for i in value:
                    if i not in column_names:
                        sc = scatter_plots(
                            df,
                            dimensionality,
                            (i, column_names[0], column_names[1]),
                            (i, column_names[0], column_names[1]),
                            category_column_name,
                            custom_dashboard,
                            is_image=is_image,
                        )
            elif len(column_names) == 1:
                value.remove(column_names[0])
                comb = combinations(value, dimensionality - 1)
                for i in list(comb):
                    sc = scatter_plots(
                        df,
                        dimensionality,
                        (i[0], i[1], column_names[0]),
                        (i[0], i[1], column_names[0]),
                        category_column_name,
                        custom_dashboard,
                        is_image=is_image,
                    )
            else:
                comb = combinations(value, dimensionality)
                for i in list(comb):
                    sc = scatter_plots(
                        df,
                        dimensionality,
                        i,
                        i,
                        category_column_name,
                        custom_dashboard,
                        is_image=is_image,
                    )
        else:
            raise NotImplementedError
        return sc

    def line_plot(df, clmn, custom_dashboard, **kwargs):
        category_column_name = kwargs["category_column_name"]
        is_image = kwargs["is_image"]
        if clmn == "":
            column_names = list()
        else:
            column_names = clmn.split(kwargs["dataframe_delimiter"])
        all_column = kwargs["all_column"]
        assert (
            len(column_names) + (1 if all_column else 0)
        ) <= 2, "Incorrect No. of columns provided"
        assert len(column_names) == 2 or all_column, "Incorrect No. of columns provided"
        value = df.columns.tolist()
        if category_column_name is not None:
            value.remove(category_column_name)

        if not all_column:
            lp = line_plots(
                df,
                column_names,
                category_column_name,
                custom_dashboard,
                is_image=is_image,
            )
        elif len(column_names) == 1:
            for i in value:
                if i not in column_names:
                    lp = line_plots(
                        df,
                        (i, column_names[0]),
                        category_column_name,
                        custom_dashboard,
                        is_image=is_image,
                    )
        else:
            comb = combinations(value, 2)
            for i in list(comb):
                lp = line_plots(
                    df, i, category_column_name, custom_dashboard, is_image=is_image
                )
        return lp
