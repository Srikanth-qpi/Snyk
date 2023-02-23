import json
import os
import shutil
import time

import cv2
import numpy as np

__all__ = ["LogWriter"]


class LogWriter(object):
    def __init__(
        self,
        logdir="logs",
        filename=None,
        microservice="modelgen",
        tag=None,
        tag_info_dict=dict(),
    ):
        self.tic = time.time()
        logdir = os.path.join(
            os.environ.get("LOGS_PATH_DIR", "."),
            logdir + "_" + os.environ.get("REQUEST_ID", "null"),
        )
        if os.path.exists(logdir):
            shutil.rmtree(logdir, ignore_errors=True)
        os.mkdir(logdir)
        self.logdir = logdir
        self.filename = (
            "custom_events_" + str(int(self.tic * 1000)) + ".log"
            if filename is None
            else filename
        )
        self.microservice = microservice
        self.tag = tag
        self.tag_info_dict = tag_info_dict
        if self.tag is not None:
            if not os.path.exists(os.path.join(self.logdir, self.tag)):
                os.mkdir(os.path.join(self.logdir, self.tag))
            if bool(self.tag_info_dict):
                if not os.path.exists(
                    os.path.join(self.logdir, self.tag, "config.json")
                ):
                    with open(
                        os.path.join(self.logdir, self.tag, "config.json"), "w"
                    ) as outfile:
                        json.dump(self.tag_info_dict, outfile)

    def add_scalar(
        self,
        values,
        tag=None,
        subtag=None,
        filename=None,
        tag_info_dict=dict(),
        subtag_info_dict=dict(),
    ):
        tag = self.tag if tag is None else tag
        tag_info_dict = self.tag_info_dict if not bool(tag_info_dict) else tag_info_dict
        if not os.path.exists(os.path.join(self.logdir, tag)):
            try:
                os.mkdir(os.path.join(self.logdir, tag))
            except:
                pass
        if not os.path.exists(os.path.join(self.logdir, tag, "config.json")):
            try:
                with open(
                    os.path.join(self.logdir, tag, "config.json"), "w"
                ) as outfile:
                    json.dump(tag_info_dict, outfile)
            except:
                pass

        subtag = self.microservice if subtag is None else subtag
        if not os.path.exists(os.path.join(self.logdir, tag, subtag)):
            os.mkdir(os.path.join(self.logdir, tag, subtag))
        if not os.path.exists(os.path.join(self.logdir, tag, subtag, "config.json")):
            with open(
                os.path.join(self.logdir, tag, subtag, "config.json"), "w"
            ) as outfile:
                json.dump(subtag_info_dict, outfile)

        with open(os.path.join(self.logdir, tag, "config.json"), "r") as inpfile:
            tag_data = json.load(inpfile)
        with open(
            os.path.join(self.logdir, tag, subtag, "config.json"), "r"
        ) as inpfile:
            subtag_data = json.load(inpfile)

        x_label = "x"
        if "x" in tag_data.keys():
            x_label = tag_data["x"].get("value", tag_data["x"].get("label", "x"))
        if filename is None:
            if "x" in subtag_data["graphs"]["graph_id_1"].keys():
                if "value" in subtag_data["graphs"]["graph_id_1"]["x"].keys():
                    x_label = subtag_data["graphs"]["graph_id_1"]["x"]["value"]
                elif x_label == "x" or x_label == tag_data["x"]["label"]:
                    x_label = subtag_data["graphs"]["graph_id_1"]["x"].get(
                        "label", x_label
                    )
        else:
            for key in subtag_data["graphs"].keys():
                if subtag_data["graphs"][key]["filename"] == filename:
                    break
            if "x" in subtag_data["graphs"][key].keys():
                if "value" in subtag_data["graphs"][key]["x"].keys():
                    x_label = subtag_data["graphs"][key]["x"]["value"]
                elif x_label == "x" or x_label == tag_data["x"]["label"]:
                    x_label = subtag_data["graphs"][key]["x"].get("label", x_label)

        y_label = "y"
        if "y" in tag_data.keys():
            y_label = tag_data["y"].get("value", tag_data["y"].get("label", "y"))
        if filename is None:
            if "y" in subtag_data["graphs"]["graph_id_1"].keys():
                if "value" in subtag_data["graphs"]["graph_id_1"]["y"].keys():
                    y_label = subtag_data["graphs"]["graph_id_1"]["y"]["value"]
                elif y_label == "y" or y_label == tag_data["y"]["label"]:
                    y_label = subtag_data["graphs"]["graph_id_1"]["y"].get(
                        "label", y_label
                    )
        else:
            for key in subtag_data["graphs"].keys():
                if subtag_data["graphs"][key]["filename"] == filename:
                    break
            if "y" in subtag_data["graphs"][key].keys():
                if "value" in subtag_data["graphs"][key]["y"].keys():
                    y_label = subtag_data["graphs"][key]["y"]["value"]
                elif y_label == "y" or y_label == tag_data["y"]["label"]:
                    y_label = subtag_data["graphs"][key]["y"].get("label", y_label)

        z_label = "z"
        if "z" in tag_data.keys():
            z_label = tag_data["z"].get("value", tag_data["z"].get("label", "z"))
        if filename is None:
            if "z" in subtag_data["graphs"]["graph_id_1"].keys():
                if "value" in subtag_data["graphs"]["graph_id_1"]["z"].keys():
                    z_label = subtag_data["graphs"]["graph_id_1"]["z"]["value"]
                elif z_label == "z" or z_label == tag_data["z"]["label"]:
                    z_label = subtag_data["graphs"]["graph_id_1"]["z"].get(
                        "label", z_label
                    )
        else:
            for key in subtag_data["graphs"].keys():
                if subtag_data["graphs"][key]["filename"] == filename:
                    break
            if "z" in subtag_data["graphs"][key].keys():
                if "value" in subtag_data["graphs"][key]["z"].keys():
                    z_label = subtag_data["graphs"][key]["z"]["value"]
                elif z_label == "z" or z_label == tag_data["z"]["label"]:
                    z_label = subtag_data["graphs"][key]["z"].get("label", z_label)

        wall_time = time.time()
        if isinstance(values, tuple):
            values = [values]
        filename = self.filename if filename is None else filename

        for value in values:
            if len(value) == 1:
                events_dict = {
                    "wall_time": wall_time,
                    "tag": tag,
                    x_label: float(value[0]),
                }
            elif len(value) == 2:
                events_dict = {
                    "wall_time": wall_time,
                    "tag": tag,
                    x_label: float(value[0]),
                    y_label: float(value[1]),
                }
            elif len(value) == 3:
                events_dict = {
                    "wall_time": wall_time,
                    "tag": tag,
                    x_label: float(value[0]),
                    y_label: float(value[1]),
                    z_label: float(value[2]),
                }
            else:
                raise NotImplementedError

            with open(os.path.join(self.logdir, tag, subtag, filename), "a") as outfile:
                json.dump(events_dict, outfile)
                outfile.write("\n")

    def add_image(
        self,
        img_obj,
        tag=None,
        subtag=None,
        filename=None,
        tag_info_dict=dict(),
        subtag_info_dict=dict(),
    ):
        tag = self.tag if tag is None else tag
        if not os.path.exists(os.path.join(self.logdir, tag)):
            try:
                os.mkdir(os.path.join(self.logdir, tag))
            except:
                pass
        if not os.path.exists(os.path.join(self.logdir, tag, "config.json")):
            try:
                with open(
                    os.path.join(self.logdir, tag, "config.json"), "w"
                ) as outfile:
                    json.dump(tag_info_dict, outfile)
            except:
                pass

        subtag = self.microservice if subtag is None else subtag
        if not os.path.exists(os.path.join(self.logdir, tag, subtag)):
            os.mkdir(os.path.join(self.logdir, tag, subtag))
        if not os.path.exists(os.path.join(self.logdir, tag, subtag, "config.json")):
            with open(
                os.path.join(self.logdir, tag, subtag, "config.json"), "w"
            ) as outfile:
                json.dump(subtag_info_dict, outfile)

        filename = self.filename if filename is None else filename

        if filename[-4:] == ".log":
            filename = filename[:-4] + ".png"
        cv2.imwrite(os.path.join(self.logdir, tag, subtag, filename), img_obj)

    def add_histogram(
        self,
        value,
        tag=None,
        subtag=None,
        filename=None,
        bins=5,
        tag_info_dict=dict(),
        subtag_info_dict=dict(),
        density=False,
        normalize=False,
        class_name=None,
    ):
        tag = self.tag if tag is None else tag
        if not os.path.exists(os.path.join(self.logdir, tag)):
            try:
                os.mkdir(os.path.join(self.logdir, tag))
            except:
                pass
        if not os.path.exists(os.path.join(self.logdir, tag, "config.json")):
            try:
                with open(
                    os.path.join(self.logdir, tag, "config.json"), "w"
                ) as outfile:
                    json.dump(tag_info_dict, outfile)
            except:
                pass

        subtag = self.microservice if subtag is None else subtag
        if not os.path.exists(os.path.join(self.logdir, tag, subtag)):
            os.mkdir(os.path.join(self.logdir, tag, subtag))
        if not os.path.exists(os.path.join(self.logdir, tag, subtag, "config.json")):
            with open(
                os.path.join(self.logdir, tag, subtag, "config.json"), "w"
            ) as outfile:
                json.dump(subtag_info_dict, outfile)

        min_value = value.min()
        max_value = value.max()
        sum_value = np.sum(value)
        sq_sum_value = np.sum(value * value)

        values = value.reshape(-1)
        num_value = len(values)
        bucket, limits = np.histogram(values, bins=bins, density=density)
        bucket_limit = limits[1:]
        bucket_limit = bucket_limit.tolist()
        bucket = bucket / np.sum(bucket) if normalize else bucket
        if class_name is not None:
            bucket_limit = [class_name]
        wall_time = time.time()
        events_dict = {
            "wall_time": wall_time,
            "tag": tag,
            "min": float(min_value),
            "max": float(max_value),
            "num": float(num_value),
            "sum": float(sum_value),
            "sum_squares": float(sq_sum_value),
            "bucket_limit": bucket_limit,
            "bucket": bucket.tolist(),
        }

        filename = self.filename if filename is None else filename

        with open(os.path.join(self.logdir, tag, subtag, filename), "a") as outfile:
            json.dump(events_dict, outfile)
            outfile.write("\n")

    def add_boxplot(
        self,
        value,
        tag=None,
        subtag=None,
        filename=None,
        tag_info_dict=dict(),
        subtag_info_dict=dict(),
    ):
        tag = self.tag if tag is None else tag
        tag_info_dict = self.tag_info_dict if not bool(tag_info_dict) else tag_info_dict
        if not os.path.exists(os.path.join(self.logdir, tag)):
            try:
                os.mkdir(os.path.join(self.logdir, tag))
            except:
                pass
        if not os.path.exists(os.path.join(self.logdir, tag, "config.json")):
            try:
                with open(
                    os.path.join(self.logdir, tag, "config.json"), "w"
                ) as outfile:
                    json.dump(tag_info_dict, outfile)
            except:
                pass

        subtag = self.microservice if subtag is None else subtag
        if not os.path.exists(os.path.join(self.logdir, tag, subtag)):
            os.mkdir(os.path.join(self.logdir, tag, subtag))
        if not os.path.exists(os.path.join(self.logdir, tag, subtag, "config.json")):
            with open(
                os.path.join(self.logdir, tag, subtag, "config.json"), "w"
            ) as outfile:
                json.dump(subtag_info_dict, outfile)
        value = np.array(value)
        min_value = float(value.min())
        max_value = float(value.max())
        t5th_perct = float(np.percentile(value, 25))
        f0th_perct = float(np.percentile(value, 50))
        s5th_perct = float(np.percentile(value, 75))

        wall_time = time.time()
        events_dict = {
            "wall_time": wall_time,
            "tag": tag,
            "min": min_value,
            "25_perct": t5th_perct,
            "50_perct": f0th_perct,
            "75_perct": s5th_perct,
            "max": max_value,
        }

        filename = self.filename if filename is None else filename

        with open(os.path.join(self.logdir, tag, subtag, filename), "a") as outfile:
            json.dump(events_dict, outfile)
            outfile.write("\n")
