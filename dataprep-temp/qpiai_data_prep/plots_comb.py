import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

__all__ = [
    "scatter_plots",
    "quantile_plots",
    "countplot",
    "distplot",
    "histogram",
    "box_plot",
    "Spearman_Correlation",
    "Pearson_Correlation",
    "line_plots",
    "barplot",
]


def Step_Size(dataframe, column_name):
    val = np.array(dataframe[column_name])
    min_val = int(np.floor(val.min() / 10) * 10)
    max_val = int(np.ceil(val.max() / 10) * 10)
    step_size = int(np.round((max_val - min_val) / 10, decimals=0))
    return min_val, max_val, step_size


def scatter_plots(
    dataframe,
    dimensionality,
    column_names,
    labels,
    target_column_name,
    dashboard_logger,
    is_image=False,
):
    assert len(column_names) == len(labels), "Inconsistent no. of columns and labels"
    assert (
        len(column_names) == dimensionality
    ), "Inconsistent no. of columns and dimensionality provided"
    assert dimensionality in [1, 2, 3], "Invalid dimensionality provided"

    if is_image:
        sns.set_style("whitegrid")
        if target_column_name is not None:
            if dimensionality == 1:
                sns.FacetGrid(dataframe, hue=target_column_name, size=4).map(
                    plt.scatter, column_names[0]
                ).add_legend()
            elif dimensionality == 2:
                sns.FacetGrid(dataframe, hue=target_column_name, size=4).map(
                    plt.scatter, column_names[0], column_names[1]
                ).add_legend()
            elif dimensionality == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.set_xlabel(column_names[0])
                ax.set_ylabel(column_names[1])
                ax.set_zlabel(column_names[2])
                unique = dataframe[target_column_name].unique().tolist()
                # ax.scatter(c=range(len(unique)))
                for i in range(len(unique)):
                    df = dataframe[dataframe[target_column_name] == unique[i]]
                    ax.scatter(
                        df[column_names[0]],
                        df[column_names[1]],
                        df[column_names[2]],
                        label=unique[i],
                    )
                plt.legend()
            else:
                raise NotImplementedError
        else:
            if dimensionality == 1:
                sns.FacetGrid(dataframe, size=4).map(
                    plt.scatter, column_names[0]
                ).add_legend()
            elif dimensionality == 2:
                sns.FacetGrid(dataframe, size=4).map(
                    plt.scatter, column_names[0], column_names[1]
                ).add_legend()
            elif dimensionality == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    dataframe[column_names[0]],
                    dataframe[column_names[1]],
                    dataframe[column_names[2]],
                )
                ax.set_xlabel(column_names[0])
                ax.set_ylabel(column_names[1])
                ax.set_zlabel(column_names[2])
            else:
                raise NotImplementedError

        tag = str(dimensionality) + "D Scatter Plots"
        subtag = ""
        for i in range(dimensionality):
            subtag = subtag + labels[i]
            if i != (dimensionality - 1):
                subtag = subtag + " vs. "

        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}

        plt.savefig("plots.png")
        img_obj = cv2.imread("plots.png")
        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )
        return

    tag = str(dimensionality) + "D Scatter Plots"

    subtag = ""
    for i in range(dimensionality):
        subtag = subtag + labels[i]
        if i != (dimensionality - 1):
            subtag = subtag + " vs. "

    tag_info_dict = {"graph_type": "scatter_plot", "merge": False}
    dimensions = ["x", "y", "z"]

    subtag_info_dict = {"graphs": {}}
    if target_column_name is not None:
        unique = dataframe[target_column_name].unique().tolist()
        for i in range(len(unique)):
            subtag_info_dict["graphs"].update(
                {
                    "graph_id_"
                    + str(i + 1): {
                        "filename": "custom_events_" + str(unique[i]) + ".log",
                        "legend_dict": {"Class": unique[i]},
                    }
                }
            )
            for j in range(dimensionality):
                min_val, max_val, step_size = Step_Size(dataframe, column_names[j])
                subtag_info_dict["graphs"]["graph_id_" + str(i + 1)].update(
                    {
                        dimensions[j]: {
                            "label": labels[j],
                            "value": column_names[j],
                            "min": min_val,
                            "max": max_val,
                            "step_size": step_size,
                        }
                    }
                )
    else:
        subtag_info_dict["graphs"].update(
            {"graph_id_1": {"filename": dashboard_logger.filename, "legend_dict": {}}}
        )
        for i in range(dimensionality):
            min_val, max_val, step_size = Step_Size(dataframe, column_names[i])
            subtag_info_dict["graphs"]["graph_id_1"].update(
                {
                    dimensions[i]: {
                        "label": labels[i],
                        "value": column_names[i],
                        "min": min_val,
                        "max": max_val,
                        "step_size": step_size,
                    }
                }
            )

    for i in range(len(dataframe)):
        if dimensionality == 1:
            value = (dataframe.at[i, column_names[0]],)
        elif dimensionality == 2:
            value = (dataframe.at[i, column_names[0]], dataframe.at[i, column_names[1]])
        elif dimensionality == 3:
            value = (
                dataframe.at[i, column_names[0]],
                dataframe.at[i, column_names[1]],
                dataframe.at[i, column_names[2]],
            )
        else:
            raise NotImplementedError

        dashboard_logger.add_scalar(
            values=value,
            tag=tag,
            subtag=subtag,
            filename="custom_events_"
            + str(dataframe.at[i, target_column_name])
            + ".log"
            if target_column_name is not None
            else dashboard_logger.filename,
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )


def line_plots(
    dataframe, column_names, target_column_name, dashboard_logger, is_image=True
):
    if is_image:
        sns.lineplot(
            data=dataframe, x=column_names[0], y=column_names[1], hue=target_column_name
        )
        plt.savefig("plots.png")
        img_obj = cv2.imread("plots.png")

        tag = "Line Plots"
        subtag = column_names[0] + " vs. " + column_names[1]

        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}

        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )

        return
    tag = "Line Plots"
    subtag = column_names[0] + " vs. " + column_names[1]

    min_val_0, max_val_0, step_size_0 = Step_Size(dataframe, column_names[0])
    min_val_1, max_val_1, step_size_1 = Step_Size(dataframe, column_names[1])

    tag_info_dict = {
        "graph_type": "line_plot",
        "merge": False,
        "x": {
            "label": column_names[0],
            "value": column_names[0],
            "min": min_val_0,
            "max": max_val_0,
            "step_size": step_size_0,
        },
        "y": {
            "label": column_names[1],
            "value": column_names[1],
            "min": min_val_1,
            "max": max_val_1,
            "step_size": step_size_1,
        },
    }

    subtag_info_dict = {"graphs": {}}
    if target_column_name is not None:
        unique = dataframe[target_column_name].unique().tolist()
        for i in range(len(unique)):
            subtag_info_dict["graphs"].update(
                {
                    "graph_id_"
                    + str(i + 1): {
                        "filename": "custom_events_" + str(unique[i]) + ".log",
                        "legend_dict": {"Class": unique[i]},
                    }
                }
            )
    else:
        subtag_info_dict["graphs"].update(
            {"graph_id_1": {"filename": dashboard_logger.filename, "legend_dict": {}}}
        )

    for i in range(len(dataframe)):
        value = (dataframe.at[i, column_names[0]], dataframe.at[i, column_names[1]])
        dashboard_logger.add_scalar(
            values=value,
            tag=tag,
            subtag=subtag,
            filename="custom_events_"
            + str(dataframe.at[i, target_column_name])
            + ".log"
            if target_column_name is not None
            else dashboard_logger.filename,
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )


def quantile_plots(dataframe, column_name, dashboard_logger, is_image=False):
    if is_image:
        res = stats.probplot(dataframe[column_name], plot=plt)
        tag = "Quantile Plot"
        subtag = "Quantile View"
        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}

        plt.savefig("plots.png")
        img_obj = cv2.imread("plots.png")
        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )
        return
    tag = "Quantile-Quantile Plot"
    tag_info_dict = {"graph_type": "quantile-quantile plot", "merge": False}
    total_len = len(dataframe[column_name])
    y_points = dataframe[column_name].tolist()
    min_val, max_val, step_size = Step_Size(dataframe, column_name)
    x_points = np.random.normal(0, 1, total_len)
    y_points = sorted(y_points)
    x_points = sorted(x_points)
    tag_info_dict.update(
        {
            "x": {
                "label": "Theoretical Values",
                "value": "Theoretical Values",
                "min": min_val,
                "max": max_val,
                "step_size": step_size,
            },
            "y": {
                "label": "Ordered Values",
                "value": "Ordered Values",
                "min": y_points[0],
                "max": y_points[-1],
                "step_size": int(
                    np.round((y_points[-1] - y_points[0]) / 10, decimals=0)
                ),
            },
        }
    )
    subtag_info_dict = {
        "graphs": {
            "graph_id_1": {
                "graph_type": "scatter_plot",
                "filename": "custom_events_scatter.log",
                "legend_dict": {"Actual Distribution": "Distribution of column"},
            },
            "graph_id_2": {
                "graph_type": "line_plot",
                "filename": "custom_events_line.log",
                "legend_dict": {"Needed Distibution": "Normal Distribution"},
            },
        }
    }
    # subtag_dict_1 = {"graph_type":"scatter plot","legend_dict":{"Actual Distribution":"Distribution of column"}}
    # subtag_dict_2 = {"graph_type":"line plot","legend_dict":{"Needed Distibution":"Normal Distribution"}}

    # if total_len %100 != 0:
    #     new_length = total_len - (total_len%100)
    #     for i in range(total_len-new_length):
    #         y_points.pop(random.randrange(len(y_points)))

    percentile_val = len(y_points) / 100
    ceil_val = math.ceil(percentile_val)
    floor_val = math.floor(percentile_val)
    dist1 = abs(percentile_val - ceil_val)
    dist2 = abs(percentile_val - floor_val)
    if dist1 <= dist2:
        percentile_val = ceil_val
    else:
        percentile_val = floor_val
    if percentile_val == 0:
        percentile_val = percentile_val + 1
    slope, intercept, r, prob, _ = stats.linregress(x_points, y_points)
    for i in range(len(y_points)):
        if i % percentile_val == 0:
            val1 = (x_points[i], y_points[i])
            dashboard_logger.add_scalar(
                values=val1,
                tag=tag,
                subtag="Quantile View",
                tag_info_dict=tag_info_dict,
                filename="custom_events_scatter.log",
                subtag_info_dict=subtag_info_dict,
            )

            val2 = (x_points[i], slope * x_points[i] + intercept)
            dashboard_logger.add_scalar(
                values=val2,
                tag=tag,
                subtag="Quantile View",
                tag_info_dict=tag_info_dict,
                filename="custom_events_line.log",
                subtag_info_dict=subtag_info_dict,
            )


def countplot(dataframe, column_name, dashboard_logger, is_image=False):
    if is_image:
        sns.countplot(data=dataframe, x=column_name)
        tag = "CountPlot"
        subtag = "Histogram View"

        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}

        plt.savefig("plots.png")
        img_obj = cv2.imread("plots.png")
        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )
        return

    def remove_na(vector):
        return vector[pd.notnull(vector)]

    a = remove_na(dataframe[column_name])

    ax = remove_na(dataframe[column_name])
    val = ax.value_counts()
    min_val = 0
    max_val = int(np.ceil(val.max() / 10) * 10)
    step_size = int(np.round((max_val - min_val) / 10, decimals=0))

    tag = "CountPlot"
    tag_info_dict = {"graph_type": "bar_plot", "merge": False}
    tag_info_dict.update(
        {
            "x": {"label": "Class", "value": "Class"},
            "y": {
                "label": "Count",
                "value": "Count",
                "min": min_val,
                "max": max_val,
                "step_size": step_size,
            },
        }
    )
    l = list(np.unique(a))
    a = a.value_counts()
    new_dict = {}
    for i in l:
        new_dict[i] = a[i]

    subtag_info_dict = {"graphs": {}}
    for i in range(len(l)):
        subtag_info_dict["graphs"].update(
            {
                "graph_id_"
                + str(i + 1): {
                    "filename": "custom_events_" + str(l[i]) + ".log",
                    "legend_dict": {"Class": str(l[i])},
                }
            }
        )

    # subtag_dict_1= {"graph_type":"Countplot","legend_dict":{"Class":l}}
    # val = []
    # for i in range(len(a)):
    #    val.append(a[i])
    for i in range(len(l)):
        dashboard_logger.add_histogram(
            value=i * np.ones(a[l[i]]),
            tag=tag,
            bins=1,
            filename="custom_events_" + str(l[i]) + ".log",
            subtag="Histogram View",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
            class_name=str(l[i]),
        )


def barplot(
    dataframe, column_names, dashboard_logger, task="missing_values", is_image=False
):
    if is_image:
        df_na = (dataframe.isnull().sum() / len(dataframe)) * 100
        for column in list(df_na.index):
            if column not in column_names:
                df_na = df_na.drop(index=column)
        sns.barplot(x=df_na.index, y=df_na)
        plt.savefig("plots.png")
        img_obj = cv2.imread("plots.png")
        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}
        tag = "Bar Plot"
        subtag = task
        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )
        return

    df_na = (dataframe.isnull().sum() / len(dataframe)) * 100
    for column in list(df_na.index):
        if int(round(df_na[column])) == 0:
            df_na = df_na.drop(index=column)
            try:
                column_names.remove(column)
            except:
                pass
    tag = "Bar Plot"
    subtag = task
    tag_info_dict = {"graph_type": "bar_plot", "merge": False}
    if len(column_names) > 0:
        max_val = df_na[column_names[0]]
        min_val = df_na[column_names[0]]
        for column_name in column_names:
            max_val = max(max_val, df_na[column_name])
            min_val = min(min_val, df_na[column_name])

        min_val = int(np.floor(min_val / 10) * 10)
        max_val = int(np.ceil(max_val / 10) * 10)
        step_size = int(np.round((max_val - min_val) / 10, decimals=0))
        tag_info_dict.update(
            {"y": {"min": min_val, "max": max_val, "step_size": step_size}}
        )

    subtag_info_dict = {"graphs": {}}
    for i in range(len(column_names)):
        subtag_info_dict["graphs"].update(
            {
                "graph_id_"
                + str(i + 1): {
                    "filename": "custom_events_" + column_names[i] + ".log",
                    "legend_dict": {"Column_names": column_names[i]},
                }
            }
        )

    for i in range(len(column_names)):
        dashboard_logger.add_histogram(
            value=i * np.ones(int(round(df_na[column_names[i]]))),
            tag=tag,
            bins=1,
            filename="custom_events_" + column_names[i] + ".log",
            subtag=subtag,
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
            class_name=column_names[i],
        )


def distplot(
    dataframe, column_name, dashboard_logger, is_image=False, hist=True, kde=True
):

    if is_image:
        sns.displot(dataframe[column_name], kde=True)
        tag = "DistPlot"
        subtag = "Distplot View"

        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}

        plt.savefig("plots.png")
        img_obj = cv2.imread("plots.png")
        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )
        return

    if dataframe[column_name] is not None:
        a = dataframe[column_name]

    # Make a a 1-d float array
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a.squeeze()

    def remove_na(vector):
        return vector[pd.notnull(vector)]

    # Drop null values from array
    a = remove_na(a)

    def iqr(a):
        a = np.asarray(a)
        q1 = stats.scoreatpercentile(a, 25)
        q3 = stats.scoreatpercentile(a, 75)
        return q3 - q1

    def freedman_diaconis_bins(a):

        a = np.asarray(a)
        if len(a) < 2:
            return 1
        h = 2 * stats.iqr(a) / (len(a) ** (1 / 3))
        # fall back to sqrt(a) bins if iqr is 0
        if h == 0:
            return int(np.sqrt(a.size))
        else:
            return int(np.ceil((a.max() - a.min()) / h))

    y = list(a)
    binc = min(freedman_diaconis_bins(a), 50)

    hist_info = np.histogram(a, bins=binc, density=True)
    val = hist_info[0]
    min_val = 0  # val.min()
    max_val = val.max()
    step_size = (max_val - min_val) / 10

    if kde:
        ax = stats.gaussian_kde(np.array(a))
        x = ax(a)

    min_y = 0
    max_y = x.max()

    tag = "DistPlot"
    tag_info_dict = {"graph_type": "DistPlot", "merge": False}
    tag_info_dict.update(
        {
            "x": {"label": column_name, "value": column_name},
            "y": {"label": "Density", "value": "Density"},
        }
    )
    subtag_dict_1 = {
        "graphs": {
            "graph_id_1": {
                "graph_type": "histogram_plot",
                "filename": "custom_events_histogram.log",
                "legend_dict": {"graph_type": "Histogram"},
                "y": {"min": min_val, "max": max_val, "step_size": step_size},
            },
            "graph_id_2": {
                "graph_type": "line_plot",
                "filename": "custom_events_line.log",
                "legend_dict": {"graph_type": "KDE_Density"},
                "y": {"min": min_y, "max": max_y},
            },
        }
    }

    dashboard_logger.add_histogram(
        value=a,
        tag=tag,
        bins=binc,
        subtag="Histogram View",
        filename="custom_events_histogram.log",
        tag_info_dict=tag_info_dict,
        subtag_info_dict=subtag_dict_1,
        density=True,
    )
    for i in range(len(y)):
        dashboard_logger.add_scalar(
            values=(y[i], x[i]),
            tag=tag,
            subtag="Histogram View",
            filename="custom_events_line.log",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_dict_1,
        )


def box_plot(dataframe, column_name, dashboard_logger, is_image=False):
    if is_image:
        subtag_info_dict = {"graphs": {}}
        for i in range(len(column_name)):
            plt.figure(i)
            sns.boxplot(x=dataframe[column_name[i]])
            tag = "Box_Plot"
            subtag = column_name[i]

            tag_info_dict = {"graph_type": "image", "merge": False}
            subtag_info_dict["graphs"].update(
                {"graph_id_" + str(i + 1): {"filename": "plots.png"}}
            )

            plt.savefig("plots.png")
            img_obj = cv2.imread("plots.png")
            dashboard_logger.add_image(
                img_obj=img_obj,
                tag=tag,
                subtag=subtag,
                filename="plots.png",
                tag_info_dict=tag_info_dict,
                subtag_info_dict=subtag_info_dict,
            )
        return
    tag = "Box_Plot"
    subtag = "Box_View"
    tag_info_dict = {"graph_type": "box plot", "merge": False}
    subtag_info_dict = {"graphs": {}}

    for i in range(len(column_name)):
        min_val, max_val, step_size = Step_Size(dataframe, column_name[i])
        tag_info_dict.update(
            {
                "x": {"label": column_name[i], "value": column_name[i]},
                "y": {
                    "label": "Value",
                    "value": "value",
                    "min": min_val,
                    "max": max_val,
                    "step_size": step_size,
                },
            }
        )
        subtag_info_dict["graphs"].update(
            {
                "graph_id_"
                + str(i + 1): {
                    "filename": "custom_events_" + column_name[i] + ".log",
                    "legend_dict": {"Column_Name": column_name[i]},
                }
            }
        )

    for i in column_name:
        dashboard_logger.add_boxplot(
            value=list(dataframe[i]),
            tag=tag,
            subtag=subtag,
            filename="custom_events_" + i + ".log",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )


def histogram(dataframe, column_name, dashboard_logger, bins=None, is_image=False):
    if is_image:
        plt.hist(x=dataframe[column_name], bins="auto" if bins is None else bins)
        plt.savefig("plots.png")
        img_obj = cv2.imread("plots.png")

        tag = "Histogram"
        subtag = column_name
        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}

        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )
        return

    tag = "Histogram"
    subtag = column_name

    tag_info_dict = {
        "graph_type": "histogram_plot",
        "x": {"label": column_name},
        "y": {"label": "Frequency"},
        "merge": False,
    }
    subtag_info_dict = {
        "graphs": {
            "graph_id_1": {"filename": dashboard_logger.filename, "legend_dict": {}}
        }
    }

    if bins is None:
        hist, _ = np.histogram(dataframe[column_name])
        bins = len(hist)
    else:
        hist, _ = np.histogram(dataframe[column_name], bins=bins)

    tag_info_dict["y"].update(
        {
            "min": int(hist.min()),
            "max": int(hist.max()),
            "step_size": int(np.round((hist.max() - hist.min()) / 10, decimals=0)),
        }
    )

    dashboard_logger.add_histogram(
        value=dataframe[column_name].to_numpy(),
        bins=bins,
        tag=tag,
        subtag=subtag,
        tag_info_dict=tag_info_dict,
        subtag_info_dict=subtag_info_dict,
    )


def Spearman_Correlation(dataframe, dashboard_logger, is_image=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    X = dataframe.select_dtypes(include=numerics)
    X.fillna(X.mean())
    if is_image:
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title("Spearman Correlation of Features", y=1.05, size=15)
        sns.heatmap(
            X.astype(float).corr(method="spearman"),
            linewidths=0.1,
            vmax=1.0,
            square=True,
            cmap=colormap,
            linecolor="white",
            annot=True,
        )
        plt.savefig("plots.png")

        tag = "Spearman"
        subtag = "Spearman View"

        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}
        img_obj = cv2.imread("plots.png")
        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )

        return


def Pearson_Correlation(dataframe, dashboard_logger, is_image=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    X = dataframe.select_dtypes(include=numerics)
    X.fillna(X.mean())
    if is_image:
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title("Pearson Correlation of Features", y=1.05, size=15)
        sns.heatmap(
            X.astype(float).corr(method="pearson"),
            linewidths=0.1,
            vmax=1.0,
            square=True,
            cmap=colormap,
            linecolor="white",
            annot=True,
        )
        plt.savefig("plots.png")

        tag = "Pearson"
        subtag = "Pearson View"

        tag_info_dict = {"graph_type": "image", "merge": False}
        subtag_info_dict = {"graphs": {"graph_id_1": {"filename": "plots.png"}}}
        img_obj = cv2.imread("plots.png")
        dashboard_logger.add_image(
            img_obj=img_obj,
            tag=tag,
            subtag=subtag,
            filename="plots.png",
            tag_info_dict=tag_info_dict,
            subtag_info_dict=subtag_info_dict,
        )

        return
