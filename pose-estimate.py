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

# Initialize streaming
lock = threading.Lock()
app = Flask(__name__)
cap = cv2.VideoCapture(0)
time.sleep(5.0)


@torch.no_grad()
def run(ip, port, anonymize=True, device='cpu', min_area=2000, thresh_val=25, yolo_conf=0.4,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
    """
    Saves mp4 result of YOLOv7 pose model and background subtraction. Main function that reads the input video
    stream and passes parameters down to the model and other functions.
    """
    device = select_device(opt.device)

    # Load model and get class names
    model = attempt_load("yolov7-w6-pose.pt", map_location=device)
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names

    if not cap.isOpened():  # check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()
    else:
        # initiate dataframe
        df = pd.DataFrame(columns=['date', 'time', 'motion', 'yolo_detections', 'bed_occupied'])

        # Frame calculations
        frame_count = 0
        total_fps = 0
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = 5
        starttime = time.monotonic()

        # Extract resizing details based of first frame
        init_background = letterbox(cap.read()[1], stride=64, auto=True)[0]
        resize_height, resize_width = init_background.shape[:2]

        # Initialize video writer
        out = None

        # Initialize video buffer for when there is no motion
        buffer_seconds = 3
        buffered_frames = collections.deque([], (fps * buffer_seconds))

        # Initialize counter for duration since last change
        static_count = 0

        # Initialize background subtraction by storing first frame to compare
        init_background_grey = background_sub_frame_prep(init_background)
        prev_grey_frame = init_background_grey.copy()

        global lock

        try:
            while cap.isOpened:
                with lock:
                    ret, cap_frame = cap.read()  # get frame and success from video capture

                    # exit if failed to get frame
                    if not ret:
                        break

                    print("Frame {} Processing".format(frame_count + 1))
                    fps_start_time = time.time()  # start time for fps calculation

                    # Background subtraction and YOLO frame prep
                    curr_grey_frame = background_sub_frame_prep(cap_frame)
                    curr_frame = yolo_frame_prep(device, cap_frame)

                    if anonymize:
                        im0 = init_background.copy()
                    else:
                        # The background will be the current frame
                        im0 = curr_frame[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c]
                        im0 = im0.cpu().numpy().astype(np.uint8)
                        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)

                    # Perform background subtraction
                    processed_frame, static_count = run_background_sub(init_background_grey, curr_grey_frame,
                                                                       prev_grey_frame, static_count, im0)
                    is_motion = processed_frame.get_is_motion

                    if is_motion:
                        # Perform YOLO. Get predictions using model
                        with torch.no_grad():
                            output_data, _ = model(curr_frame)

                        # Specifying model parameters using non max suppression
                        output_data = non_max_suppression_kpt(output_data,
                                                              opt.yolo_conf,  # Conf. Threshold.
                                                              0.4,  # IoU Threshold.
                                                              nc=model.yaml['nc'],  # Number of classes.
                                                              nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                              kpt_label=True)

                        # Place the model outputs onto an frame
                        processed_frame = yolo_output_plotter(processed_frame.get_frame, names, output_data)

                    date_time = place_txt_results(processed_frame.get_bed_occupied, is_motion,
                                                  processed_frame.get_num_detections,
                                                  processed_frame.get_frame)

                    update_df(processed_frame.get_bed_occupied, date_time, df, is_motion,
                              processed_frame.get_num_detections, frame_count, fps)

                    # Figure out how to save the frame based off buffer
                    buffer_lst = list(buffered_frames)
                    is_motion_lst = [f.get_is_motion for f in buffer_lst]
                    if not any(is_motion_lst) and is_motion:
                        curr_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                        out = cv2.VideoWriter(f"output_videos/{curr_time}.mp4",
                                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (resize_width, resize_height))
                        for f in buffer_lst:
                            out.write(f.get_frame)
                        out.write(processed_frame.get_frame)
                    elif any(is_motion_lst) and out is not None:
                        out.write(processed_frame.get_frame)
                    elif not any(is_motion_lst) and not is_motion and out is not None:
                        out.release()

                    # Should we reset the background?
                    is_person_lst = [f.get_num_detections for f in buffer_lst]
                    if static_count == fps * 10 and sum(is_person_lst) == 0:
                        init_background = letterbox(cap_frame, stride=64, auto=True)[0]
                        init_background_grey = background_sub_frame_prep(init_background)
                        static_count = 0

                    # Stream the frame
                    flag, encoded_image = cv2.imencode(".jpg", processed_frame.get_frame)
                    if not flag:
                        continue

                    # update buffer
                    buffered_frames.append(processed_frame)

                    # update the previous frame
                    prev_grey_frame = curr_grey_frame

                    # FPS calculations
                    end_time = time.time()
                    total_fps += 1 / (end_time - fps_start_time)
                    frame_count += 1

                    time.sleep((1 / fps) - ((time.monotonic() - starttime) % (1 / fps)))

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(
                    encoded_image) + b'\r\n')  # yield some text and the output frame in the byte format

        except KeyboardInterrupt:
            pass

        cap.release()
        curr_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        df.to_csv(f"output_videos/{curr_time}.csv", index=False)
        print(f"Average FPS: {total_fps / frame_count:.3f}")


def update_df(bed_occupied, date_time, df, is_motion, num_detections, frame_count, fps):
    """Updates the dataframe df with details from the current frame
    """
    if frame_count % int(fps) == 0:
        new_row = {'date': date_time.strftime("%Y-%m-%d"), 'time': date_time.strftime("%H:%M:%S"), 'motion': is_motion,
                   'yolo_detections': int(num_detections), 'bed_occupied': bool(bed_occupied)}
        df.loc[len(df)] = new_row


def place_txt_results(bed_occupied, is_motion, num_detections, processed_frame):
    """Places the text of the results onto the processed frame.
    """
    cv2.putText(processed_frame, "Motion: {}".format(is_motion), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(processed_frame, "YOLO detections: {}".format(num_detections), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(processed_frame, "Bed occupied: {}".format(bed_occupied), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # put timestamp
    dt = datetime.now()
    cv2.putText(processed_frame, dt.strftime("%Y-%m-%d %H:%M:%S"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    return dt


def run_background_sub(background_grey, curr_grey_frame, prev_grey_frame, static_count, curr_color_frame):
    """
    Returns a frame with the background subtraction completed and labeled, the current grey frame to serve as the new
    background, and the updated static counter.
    1) computes difference in the frame from the original background and previous frame
    2) thresholding to filter areas with significant change in pixel values
    3) highlights the areas with motion in white
    """
    is_motion = False
    prev_diff = False

    # compute the  difference
    frame_delta = cv2.absdiff(background_grey, curr_grey_frame)
    prev_delta = cv2.absdiff(prev_grey_frame, curr_grey_frame)

    # pixels are either 0 or 255.
    thresh = cv2.threshold(frame_delta, opt.thresh_val, 255, cv2.THRESH_BINARY)[1]
    prev_thresh = cv2.threshold(prev_delta, opt.thresh_val, 255, cv2.THRESH_BINARY)[1]

    # find the outlines of the white parts from background
    thresh = cv2.dilate(thresh, None, iterations=3)  # size of foreground increases
    curr_contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curr_contours = imutils.grab_contours(curr_contours)

    # find outlines from previous frame
    prev_thresh = cv2.dilate(prev_thresh, None, iterations=2)
    prev_contours = cv2.findContours(prev_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prev_contours = imutils.grab_contours(prev_contours)

    for c in curr_contours:
        # Only care about contour if it's larger than the min
        if cv2.contourArea(c) >= opt.min_area:
            is_motion = True
            break

    for c in prev_contours:
        if cv2.contourArea(c) >= opt.min_area:
            prev_diff = True
            static_count = 0
            break

    if not prev_diff:
        static_count += 1

    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(curr_color_frame, 0.75, thresh_color, 0.25, 0)

    overlay = frame.ProcessedFrame(overlay, is_motion)

    return overlay, static_count


def yolo_output_plotter(background, names, output_data):
    """
    Plots the yolo model outputs onto background. Calculates the number of detections and places them on the background.
    Returns the processed frame.
    """
    # if there are no poses, then there is no one on the bed
    bed_occupied = False
    n = 0

    for i, pose in enumerate(output_data):  # detections per image
        if len(output_data) and len(pose[:, 5].unique()) != 0:  # check if no pose
            for c in pose[:, 5].unique():  # Print results
                n = (pose[:, 5] == c).sum()  # detections per class
                # "YOLO detections: {}".format(n)

            for det_index, (*xyxy, conf, cls) in enumerate(
                    reversed(pose[:, :6])):  # loop over poses for drawing on frame
                c = int(cls)  # integer class
                keypoints = pose[det_index, 6:]
                label = None if opt.hide_labels else (
                    names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')

                bed_occupied = plot_one_box_kpt(xyxy, background, label=label, color=colors(c, True),
                                                line_thickness=opt.line_thickness, kpt_label=True, kpts=keypoints,
                                                steps=3,
                                                orig_shape=background.shape[:2])

    processed_frame = frame.ProcessedFrame(background, True, n, bed_occupied)

    return processed_frame


@app.route("/")  # Decorator that routes you to a specific URL: in this case just /.
def index():
    """
    Function to render the index.html template and serve up the output video stream
    """
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    Function to use the flask function Response.
    MIME type a.k.a. media type: indicates the nature and format of a document, file, or assortment of bytes. MIME
    types are defined and standardized in IETF's RFC6838.
    """
    return Response(main(opt), mimetype="multipart/x-mixed-replace; boundary=frame")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='0 for webcam or video path')  # video source
    parser.add_argument('--anonymize', action='store_true',
                        help="anonymize by return video with first frame as background")
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  # device arguments
    parser.add_argument('--min-area', default=2000, type=int,
                        help='define min area in pixels that counts as motion')
    parser.add_argument('--thresh-val', default=40, type=int,
                        help='define threshold value for difference in pixels for background subtraction')
    parser.add_argument('--yolo-conf', default=0.4, type=float,
                        help='define min confidence level for YOLO model')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')  # box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # boxhideconf
    parser.add_argument("--ip", type=str, required=True, help="ip address of the device")
    parser.add_argument("--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    opt = parser.parse_args()
    return opt


# main function
def main(options):
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device)
    main(opt)

    # start a thread that will perform motion detection
    t = threading.Thread(target=main(opt))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host='10.42.0.1', port=opt.port, debug=True, threaded=True, use_reloader=False)
