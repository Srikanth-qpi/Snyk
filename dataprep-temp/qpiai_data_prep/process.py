import traceback
import json
from pymongo import MongoClient
from pprint import pprint
import time
import datetime
import os
import shutil
from qpiai_utils.database import QpiDatabase
from qpiai_utils.errors import QpiException
from qpiai_data_prep.constants import *
from qpiai_data_prep.calc.math import addition, divide
from qpiai_data_prep.image.new_image_dataprep import image_dataprep
from qpiai_data_prep.pipe import pipeline
from qpiai_data_prep.tab.tab_dataprep import tab_dataprep
#from qpiai_data_prep.text.text_dataprep import text_dataprep
from qpiai_data_prep.video.video_dataprep import video_dataprep
from qpiai_data_prep.voice.voice_dataprep import voice_dataprep
from custom import run

from config import DB_LOG, RequestModel


def process_main(req_args):
    req_args = RequestModel(**req_args)
    print("DB_LOG: ", DB_LOG)
    db = QpiDatabase(
        request_id=req_args.request_id,
        user_id=req_args.user_id,
        feature_type=req_args.feature_type,
        input_args=req_args.input.dict(),
        no_db=not (DB_LOG),
    )
    with open('temp/db_config.json', 'w') as f:
        db_config = dict(request_id=req_args.request_id,
        user_id=req_args.user_id,
        feature_type=req_args.feature_type,
        input_args={},
        no_db=not (DB_LOG),)
        json.dump(db_config, f)

    print("Process Request ID:", req_args.request_id)

    try:
        db.update_process_state(status="running")
        output = function_wrapper(req_args)
        output = json.loads(output)
        response =  output
        print(type(response))

        db.update_output(status="success", output=response)
    except QpiException as qpi_exp:
        db.update_output(
            status="failure",
            output={"message": qpi_exp.message},
        )
        response = {"status": "error", "message": qpi_exp.message}

    except Exception as exc:
        db.update_output(
            status="error",
            output={
                "error_type": str(exc.__class__),
                "traceback": traceback.format_exc(),
            },
        )
        response = {
            "status": "error",
            "error_type": str(exc.__class__),
            "traceback": traceback.format_exc(),
        }

    print("Request Completed. ID:", req_args.request_id)
    db.close()
    print(response)
    return response


def change_output_path(response, user_id, request_id, feature_type):
    if feature_type in ['modelgen','modelpredict','dataprep','modeldeploy','kdd']:
        datafiles_mnt = os.environ.get("DATASET_MOUNT",".")
        response_new = json.loads(response)
        logs = False
        if "logsOutput" in response_new:
            log_directory = response_new.pop("logsOutput")
            logs = True
        if not bool(response_new):
            response = {"logsOutput": log_directory}
        else:
            if type(list(response_new.values())[0]) is list:
                new_list = []
                files_list = list(response_new.values())[0]
                for file in files_list:
                    filename,filextension = os.path.splitext(file)
                    new_path = os.path.join(datafiles_mnt, str(user_id), str(request_id), "datafiles", filename.split("/")[-1]+"_"+request_id + filextension)
                    shutil.copy(file,new_path)
                    new_list.append(os.path.abspath(new_path))
                response = {list(response_new.keys())[0]:new_list}
                if logs:
                    response.update({"logsOutput": os.path.abspath(log_directory)})
                return json.dumps(response)
            filename,filextension = os.path.splitext(list(response_new.values())[0])
            new_path = os.path.join(datafiles_mnt, str(user_id), str(request_id), "datafiles", filename.split("/")[-1]+"_"+request_id + filextension)
            shutil.copy(list(response_new.values())[0],new_path)
            response = {list(response_new.keys())[0]:os.path.abspath(new_path)}
        if logs:
            response.update({"logsOutput": os.path.abspath(log_directory)})
        return json.dumps(response)
    else:
        return response



def function_wrapper(req_args):
    print("mongodb://{0}:{1}@{2}/{3}".format(MONGODB_USER, MONGODB_PASS, MONGODB_HOST, MONGODB_NAME ))
    my_client = MongoClient("mongodb://{0}:{1}@{2}/{3}".format(MONGODB_USER, MONGODB_PASS, MONGODB_HOST, MONGODB_NAME ))
    #my_client = MongoClient(MONGODB_CONN_ID)
    db = my_client.qpiai
    datafiles_mnt = os.environ.get("DATASET_MOUNT","")
    request_id = req_args.request_id
    user_id = req_args.user_id
    feature_type = req_args.feature_type
    if not os.path.exists(os.path.join(datafiles_mnt, str(user_id))):
        os.mkdir(os.path.join(datafiles_mnt, str(user_id)))
    if not os.path.exists(os.path.join(datafiles_mnt, str(user_id), str(request_id))):
        os.mkdir(os.path.join(datafiles_mnt, str(user_id), str(request_id)))
    '''if not os.path.exists(os.path.join(datafiles_mnt,"datasets")):
        os.mkdir(os.path.join(datafiles_mnt,"datasets"))
    if not os.path.exists(os.path.join(datafiles_mnt,"datafiles")):
        os.mkdir(os.path.join(datafiles_mnt,"datafiles"))
    if not os.path.exists(os.path.join(datafiles_mnt,"logs")):
        os.mkdir(os.path.join(datafiles_mnt,"logs"))'''
    job_id = job_id = str(request_id + "_" + str(user_id))
    if not os.path.exists(os.path.join(datafiles_mnt, str(user_id), str(request_id), "datasets")):
        os.mkdir(os.path.join(datafiles_mnt, str(user_id), str(request_id), "datasets"))
    os.environ['DATA_PATH_DIR'] = os.path.join(datafiles_mnt, str(user_id), str(request_id), "datasets")
    if not os.path.exists(os.path.join(datafiles_mnt, str(user_id), str(request_id), "datafiles")):
        os.mkdir(os.path.join(datafiles_mnt, str(user_id), str(request_id), "datafiles"))
    if not os.path.exists(os.path.join(datafiles_mnt, str(user_id), str(request_id), "logs")):
        os.mkdir(os.path.join(datafiles_mnt, str(user_id), str(request_id), "logs"))
    os.environ['LOGS_PATH_DIR']=os.path.join(datafiles_mnt, str(user_id), str(request_id), "logs")
    os.environ['OUTPUT_PATH_DIR'] = os.path.join(datafiles_mnt, str(user_id), str(request_id), "datafiles")
    os.environ['REQUEST_ID']=request_id
    mydict = { "request_id": request_id, 
        "user_id": user_id,
        "feature_type":feature_type,
        "job_id":job_id,
        "output":"",
        "timestamp":datetime.datetime.utcnow(),
        "status":"running"
    }
    #cur = db.job_response.find({"$and":[{"request_id":request_id},{"job_id":job_id},{"feature_type":feature_type}]})
    #if cur.count() == 0:
        #print("Here I am")
        #db.Response.insert_one(mydict)
        #db.job_response.insert_one(mydict)
    #else:
        #query = {
            #"request_id": request_id,
            #"user_id": user_id,
            #"job_id": job_id
        #}
        #newvalues = { "$set": { "status": "running"}}
        #db.job_response.update(query,newvalues)
    response = qpiai_data(req_args)
    response_dict = json.loads(response)
    if any(isinstance(i,dict) for i in response_dict.values()):
        response = dict()
        for k in response_dict.keys():
            solution = change_output_path(json.dumps(response_dict[k]), user_id, request_id, feature_type)
            response.update({k:json.loads(solution)})
        response = json.dumps(response)
    else:
        response = change_output_path(response, user_id, request_id, feature_type)
    return response
        


def qpiai_data(req_args):
    kwargs = {
        "dataframe_delimiter": req_args.input.dataframe_delimiter,
        "category_column_name": req_args.input.category_column_name,
        "num_components": req_args.input.num_components,
        "perplexity": req_args.input.perplexity,
        "all_column": req_args.input.all_column,
        "is_image": req_args.input.is_image,
        "inc_data": req_args.input.inc_data,
        "shift_limit": req_args.input.shift_limit,
        "scale_limit": req_args.input.scale_limit,
        "rotate_limit": req_args.input.rotate_limit,
        "width": req_args.input.width,
        "height": req_args.input.height,
        "quality_lower": req_args.input.quality_lower,
        "quality_upper": req_args.input.quality_upper,
        "blur_limit_lower": req_args.input.blur_limit_lower,
        "blur_limit_upper": req_args.input.blur_limit_upper,
        "multi_min": req_args.input.multi_min,
        "multi_max": req_args.input.multi_max,
        "n_holes": req_args.input.n_holes,
        "max_height": req_args.input.max_height,
        "max_width": req_args.input.max_width,
        "alpha_elastic": req_args.input.alpha_elastic,
        "alpha_affine": req_args.input.alpha_affine,
        "sigma": req_args.input.sigma,
        "hue_shift_limit": req_args.input.hue_shift_limit,
        "sat_shift_limit": req_args.input.sat_shift_limit,
        "val_shift_limit": req_args.input.val_shift_limit,
        "distort_limit": req_args.input.distort_limit,
        "distort_limit_opt": req_args.input.distort_limit_opt,
        "shift_limit_opt": req_args.input.shift_limit_opt,
        "bright_limit": req_args.input.bright_limit,
        "contrast_limit": req_args.input.contrast_limit,
        "r_shift": req_args.input.r_shift,
        "b_shift": req_args.input.b_shift,
        "g_shift": req_args.input.g_shift,
        "alpha_fpca": req_args.input.alpha_fpca,
        "scale_min": req_args.input.scale_min,
        "scale_max": req_args.input.scale_max,
        "hue": req_args.input.hue,
        "saturation": req_args.input.saturation,
        "channel_drop_max": req_args.input.channel_drop_max,
        "x_min": req_args.input.x_min,
        "x_max": req_args.input.x_max,
        "y_min": req_args.input.y_min,
        "y_max": req_args.input.y_max,
        "r_mean": req_args.input.r_mean,
        "g_mean": req_args.input.g_mean,
        "b_mean": req_args.input.b_mean,
        "r_std": req_args.input.r_std,
        "b_std": req_args.input.b_std,
        "g_std": req_args.input.g_std,
        "probability": req_args.input.probability,
        "cmd_info": req_args.input.cmd_info,
    }
    
    if req_args.input.datatype == "tab":
        if req_args.input.data_prepcmd == "None":
            p = pipeline(
                req_args.input.dataset,
                req_args.input.target_device,
                req_args.input.num_device,
                **kwargs
            )
            output = p.run()
        else:
            output = tab_dataprep(
                req_args.input.dataset,
                req_args.input.target_device,
                req_args.input.num_device,
                req_args.input.data_prepcmd,
                req_args.input.clmn,
                **kwargs
            )

    elif req_args.input.datatype == "text":
        output = text_dataprep(
            req_args.input.dataset,
            req_args.input.target_device,
            req_args.input.num_device,
            req_args.input.data_prepcmd,
            req_args.input.clmn,
            **kwargs
        )

    elif req_args.input.datatype == "image":
        output = image_dataprep(
            req_args.input.dataset,
            req_args.input.dataset_format,
            req_args.input.target_device,
            req_args.input.num_device,
            req_args.input.data_prepcmd,
            **kwargs
        )

    elif req_args.input.datatype == "video":
        output = video_dataprep(
            req_args.input.dataset,
            req_args.input.target_device,
            req_args.input.num_device,
            req_args.input.data_prepcmd,
        )

    elif req_args.input.datatype == "voice":
        output = voice_dataprep(
            req_args.input.dataset,
            req_args.input.target_device,
            req_args.input.num_device,
            req_args.input.data_prepcmd,
        )

    elif req_args.input.datatype == "custom_tab" or "custom_image":
        output = run(req_args.input.zipfile, req_args.input.dataset, req_args.input.datatype )

    resultObj = output
    json_resp = json.dumps(resultObj)
    return json_resp