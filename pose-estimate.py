import argparse
import time

import cv2
import imutils
import numpy as np
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import colors, plot_one_box_kpt
from utils.torch_utils import select_device


@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source=0, anonymize=True, device='cpu',
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
    """
    Saves mp4 result of YOLOv7 pose model and background subtraction. Main function that reads the input video
    stream and passes parameters down to the model and other functions.
    """
    device = select_device(opt.device)

    # Load model and get class names
    model = attempt_load(poseweights, map_location=device)
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names

    # Check if source is webcam/camera or video file
    if source.is_integer():
        cap = cv2.VideoCapture(int(source))
        time.sleep(2.0)  # Wait for webcam to turn on
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():  # check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_count = 0
        total_fps = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # extract resizing details based of first frame
        first_frame_init = letterbox(cap.read()[1], stride=64, auto=True)[0]
        resize_height, resize_width = first_frame_init.shape[:2]

        # Initialize video writer
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"output_videos/{out_video_name}_yolo_sub.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (resize_width, resize_height))

        # Initialize background subtraction
        background = None  # stores the a frame to compare against for changes
        background_color = None
        static_counter = 0  # counts the number of frames in which the video stream hasn't changed significantly.
        prev_frame = None

        while cap.isOpened:  # loop until cap opened or video not complete
            ret, frame = cap.read()  # get frame and success from video capture

            first_frame = first_frame_init.copy()

            if ret:  # if success is true, means frame exist
                print("Frame {} Processing".format(frame_count + 1))
                start_time = time.time()  # start time for fps calculation

                image = yolo_frame_prep(device, frame)

                # Get predictions using model
                with torch.no_grad():
                    output_data, _ = model(image)

                # Specifying model parameters using non max suppression
                output_data = non_max_suppression_kpt(output_data,
                                                      0.45,  # Conf. Threshold.
                                                      0.65,  # IoU Threshold.
                                                      nc=model.yaml['nc'],  # Number of classes.
                                                      nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                      kpt_label=True)

                # output = output_to_keypoint(output_data)

                if not anonymize:
                    # The background will be the current frame
                    im0 = image[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c]
                    im0 = im0.cpu().numpy().astype(np.uint8)
                    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
                else:
                    im0 = first_frame

                # Place the model outputs onto an image
                yolo_output_plotter(im0, names, output_data)

                # Perform background substitution
                curr_grey_frame = background_sub_frame_prep(frame)
                processed_frame, background, static_counter = run_background_sub(background, curr_grey_frame, fps,
                                                                                 frame, prev_frame, im0,
                                                                                 static_counter)
                prev_frame = curr_grey_frame

                # FPS calculations
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                out.write(im0)

            else:
                break

        cap.release()
        print(f"Average FPS: {total_fps / frame_count:.3f}")


def background_sub_frame_prep(frame):
    """
    Prepares the frame to be used in background subtraction. THe frame is converted to the same size as the YOLO
    model frame and converted to grayscale with blur. The blurring ensures high frequency noise doesn't throw off
    the algorithm.
    """
    image = letterbox(frame, stride=64, auto=True)[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (27, 27), 0)
    return gray


def run_background_sub(background, curr_grey_frame, fps, frame, prev_frame, processed_frame, static_counter):
    """
    Returns a frame with the background subtraction completed and labeled, the current grey frame to serve as the new
    background, and the updated static counter.
    1) initializes background if it is the first frame
    2) computes difference in frames
    3) thresholding to filter areas with significant change in pixel values
    4) finds the outlines of areas with changes and draws a box around it
    """
    text = "Background subtraction: no motion"
    min_area = 2000
    threshold_val = 75
    static_secs = 4  # how many seconds with no change until update

    if background is None:
        background = curr_grey_frame
        background_color = frame
        prev_frame = curr_grey_frame  # if the background is None, the prev_frame is also None. Initialize it.

    # compute the  difference
    frame_delta = cv2.absdiff(background, curr_grey_frame)
    prev_delta = cv2.absdiff(prev_frame, curr_grey_frame)

    # pixels are either 0 or 255.
    thresh = cv2.threshold(frame_delta, threshold_val, 255, cv2.THRESH_BINARY)[1]
    prev_thresh = cv2.threshold(prev_delta, threshold_val, 255, cv2.THRESH_BINARY)[1]

    # find the outlines of the white parts
    thresh = cv2.dilate(thresh, None, iterations=2)  # size of foreground increases
    prev_thresh = cv2.dilate(prev_thresh, None, iterations=2)
    curr_contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prev_contours = cv2.findContours(prev_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curr_contours = imutils.grab_contours(curr_contours)
    prev_contours = imutils.grab_contours(prev_contours)

    # flag that changes to True if the current frame is significantly different from the last frame.
    background_delta = False

    # loop over the contours
    for c in curr_contours:
        # Only care about contour if it's larger than the min
        if cv2.contourArea(c) >= min_area:
            # compute the bounding box for the contour if area is large, draw it on the frame, and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "Background subtraction: motion"
    for c in prev_contours:
        # Only care about contour if it's larger than the min
        if cv2.contourArea(c) >= min_area:
            background_delta = True
            static_counter = 0  # reset to 0 since the background changed
    if not background_delta:
        static_counter += 1
        if static_counter > (fps * static_secs):
            background = curr_grey_frame
            background_color = frame
            static_counter = 0

    cv2.putText(processed_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return processed_frame, background, static_counter


def yolo_frame_prep(device, frame):
    """
    Prepares the frame for use in the YOLO model using the specified device.
    """
    orig_image = frame  # store frame
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
    image = letterbox(image, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)  # convert image data to device
    image = image.float()  # convert image to float precision (cpu)
    return image


def yolo_output_plotter(background, names, output_data):
    """
    Plots the yolo model outputs onto background. Calculates the number of detections and places them on the background.
    Returns the processed frame.
    """
    for i, pose in enumerate(output_data):  # detections per image
        if len(output_data) and len(pose[:, 5].unique()) != 0:  # check if no pose
            for c in pose[:, 5].unique():  # Print results
                n = (pose[:, 5] == c).sum()  # detections per class
                # print("No of Objects in Current Frame : {}".format(n))
                cv2.putText(background, "YOLO detections: {}".format(n), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

            for det_index, (*xyxy, conf, cls) in enumerate(
                    reversed(pose[:, :6])):  # loop over poses for drawing on frame
                c = int(cls)  # integer class
                kpts = pose[det_index, 6:]
                label = None if opt.hide_labels else (
                    names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')

                plot_one_box_kpt(xyxy, background, label=label, color=colors(c, True),
                                 line_thickness=opt.line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                 orig_shape=background.shape[:2])

        else:
            cv2.putText(background, "YOLO detections: 0", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

    return background


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='preped_videos/7855_test.mp4',
                        help='0 for webcam or video path')  # video source
    parser.add_argument('--anonymize', action=argparse.BooleanOptionalAction, default=True,
                        help="anonymize by return video with first frame as background")
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  # device arugments
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')  # box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # boxhideconf
    opt = parser.parse_args()
    return opt


# main function
def main(options):
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
