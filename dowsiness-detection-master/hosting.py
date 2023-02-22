import numpy as np
from tempfile import NamedTemporaryFile
from flask import Flask
from flask import jsonify, make_response
from flask import request
from PIL import Image
from detect_drowsiness import DrowsinessDetect
import cv2
import json
import boto3
import configparser
import sys
import os

def create_app():
    config = configparser.ConfigParser()
    config.read('config.ini')
    env = 'dev'
    app = Flask(__name__)
    DROWSINESS = DrowsinessDetect()
    client = boto3.client('s3', region_name=config[env]['aws_region_name'],
                          aws_access_key_id=config[env]['aws_access_key_id'],
                          aws_secret_access_key=config[env]['aws_secret_access_key'])


    @app.route("/drowsiness_detection", methods=["POST"])
    def drowsiness_detection():
        try:
            data = request.json
            print(data)
            img = data['image']
            bucket = data['bucket']
            client.download_file(bucket, img, img)
            drowsiness = DROWSINESS.detect(img)
            response = {'drowsiness': drowsiness}
            os.remove(img)
            print(str(response))
            return make_response(jsonify(results=response), 200)
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            response = {'error': str(sys.exc_info()[0])}
            return make_response(jsonify(results=response), 200)

    @app.route('/')
    def index():
        return 'Response is 200'

    return app
