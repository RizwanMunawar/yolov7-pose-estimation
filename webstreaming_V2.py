import argparse
import collections
import time
from datetime import datetime
import pandas as pd
import threading
import flask
from flask import Response, Flask, render_template

import cv2
import imutils
import numpy as np
import torch

from models.experimental import attempt_load
from utils import frame
from utils.frame import background_sub_frame_prep, yolo_frame_prep
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import colors, plot_one_box_kpt
from utils.torch_utils import select_device

"""
Performing some initializations
"""
lock = threading.Lock() # to prevent race condition; in this case, it ensures that one thread isn't trying to read the frame as it is being updated.
app = Flask(__name__) # initialize a flask object
camera = cv2.VideoCapture(0) # so that we can use the webcam
time.sleep(5.0) # wait for the camera to warm up. Changed to 5 seconds as it looks like it takes a while for the camera to turn on.

"""
Function to render the index.html template and serve up the output video stream
"""
@app.route("/") # Decorator that routes you to a specific URL: in this case just /.
def index():
    return render_template("index.html")

"""
Function to return a frame

@Alex: I was thinking we can run your model in here, as all this function is doing right now is reading in a frame from the camera, then transforming the output to byte format.
So, depending on whether your model does things frame by frame, maybe we can just add a step in between where we feed the current frame into your model, get an output frame, 
and then transform that output frame into byte format?
"""
def return_frame():
    # grab global references to the video stream, output frame, and lock variables.
    global camera, lock

    while True:
        with lock: # wait until the lock is acquired. We need to acquire the lock to ensure the frame variable is not accidentally being read by a client while we are trying to update it.
            success, frame = camera.read() # read the camera frame
            if not success:
                break
            else:
                yolo_prep = yolo_frame_prep(0, frame)
                (flag, encodedImage) = cv2.imencode(".jpg", yolo_prep) # encode the frame in JPEG format
                if not flag: # ensure the frame was successfully encoded
                    continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n') # yield some text and the output frame in the byte format

"""
Function to use the flask function Response
"""
@app.route("/video_feed")
def video_feed():
    return Response(return_frame(), mimetype="multipart/x-mixed-replace; boundary=frame") #MIME type a.k.a. media type: indicates the nature and format of a document, file, or assortment of bytes. MIME types are defined and standardized in IETF's RFC6838. 

"""
Handle parsing commmand line arguments and launch the Flask app
"""
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    args=vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=return_frame)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host='10.42.0.1', port=args["port"], debug=True, threaded=True, use_reloader=False) # Find the IP address of the Jetson device manually, then replace host=... with that ip address
    # to find the ip address, use the ifconfig command. Under wlan, the ip address is listed after the "inet" field.

camera.release() # Before ending the script release the camera.
print("End of script.")
