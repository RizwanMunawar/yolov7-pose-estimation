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
def run(poseweights="yolov7-w6-pose.pt", source="preped_videos/7855_test.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
    frame_count = 0  # count no of frames
    total_fps = 0  # count total fps

    # Select device
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model and get class names
    model = attempt_load(poseweights, map_location=device)
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names

    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))  # pass video to videocapture object
    else:
        cap = cv2.VideoCapture(source)  # pass video to videocapture object

    if not cap.isOpened():  # check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  # get video frame width
        frame_height = int(cap.get(4))  # get video frame height
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(fps)

        # video writer and resizing
        first_frame_init = letterbox(cap.read()[1], stride=64, auto=True)[0]
        resize_height, resize_width = first_frame_init.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"output_videos/{out_video_name}_yolo.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (resize_width, resize_height))

        # Inital steps for background subtraction
        background = None  # stores the first frame of the video file/webcam stream. Assumption is that the first frame will contain no motion and just the background.
        #                   So, we can model the background of our video stream using only the first frame of the video
        background_color = None  # Used for overlaying silhouette
        static_counter = 0  # counts the number of frames in which the video stream hasn't changed significantly. Use this to update backgrounds as needed.
        # loop over the frames of the video
        prev_frame = None  # stores the previous frame

        while cap.isOpened:  # loop until cap opened or video not complete

            print("Frame {} Processing".format(frame_count + 1))
            ret, frame = cap.read()  # get frame and success from video capture

            first_frame = first_frame_init.copy()

            if ret:  # if success is true, means frame exist
                start_time = time.time()  # start time for fps calculation
                image = yolo_frame_helper(device, frame)

                # Get predictions using model
                with torch.no_grad():
                    output_data, _ = model(image)

                # Specifying model parameters
                output_data = non_max_suppression_kpt(output_data,  # Apply non max suppression
                                                      0.45,  # Conf. Threshold.
                                                      0.65,  # IoU Threshold.
                                                      nc=model.yaml['nc'],  # Number of classes.
                                                      nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                      kpt_label=True)

                # output = output_to_keypoint(output_data)

                # only required if you want to overlay the background
                # im0 = image[0].permute(1, 2,
                #                        0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                # im0 = im0.cpu().numpy().astype(np.uint8)
                #
                # im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                # place the model outputs onto an image
                processed_frame = yolo_model_helper(first_frame, names, output_data)

                curr_grey_frame = background_sub_frame_helper(frame)
                text = "Background Subtraction: Empty"

                if background is None:
                    background = curr_grey_frame
                    background_color = frame
                    prev_frame = curr_grey_frame  # if the background is None, the prev_frame is also None. Initialize it.
                    continue
                # compute the absolute difference between the current frame and current background frame/previous frame
                frameDelta = cv2.absdiff(background,
                                         curr_grey_frame)  # is a function which helps in finding the absolute difference between the pixels of the 2 image arrays
                prevDelta = cv2.absdiff(prev_frame, curr_grey_frame)
                # threshold (thresholding is the binarization of an image. we want to convert a grayscale image to a binary image, where the pixels are either 0 or 255.)
                thresh = cv2.threshold(frameDelta, 75, 255, cv2.THRESH_BINARY)[
                    1]  # we want to threshold frameDelta to reveal regions of the image that only have significant changes in pixel intense values.
                prevThresh = cv2.threshold(prevDelta, 75, 255, cv2.THRESH_BINARY)[1]
                # find the outlines of the white parts
                thresh = cv2.dilate(thresh, None, iterations=2)  # size of foreground increases
                prevThresh = cv2.dilate(prevThresh, None, iterations=2)
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                prevCnts = cv2.findContours(prevThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                prevCnts = imutils.grab_contours(prevCnts)
                background_delta = False  # flag that changes to True if the current frame is significantly different from the last frame.
                # loop over the contours
                for c in cnts:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < 2000:
                        continue
                    # compute the bounding box for the contour if the area is lare enough, draw it on the frame, and update the text
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    text = "Background Subtraction: Occupied"
                for c in prevCnts:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < 2000:
                        continue
                    background_delta = True
                    static_counter = 0  # reset to 0 since the background changed
                    break
                if not background_delta:
                    static_counter += 1
                    if static_counter > (fps * 4):  # if the frames haven't changed in more than 10 frames, update background
                        background = curr_grey_frame
                        background_color = frame
                        static_counter = 0

                cv2.putText(processed_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                prev_frame = curr_grey_frame

                # FPS calculations
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                out.write(processed_frame)

            else:
                break

        cap.release()
        print(f"Average FPS: {total_fps / frame_count:.3f}")


def yolo_frame_helper(device, frame):
    """Prepares the frame for use in the YOLO model"""
    orig_image = frame  # store frame
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
    image = letterbox(image, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)  # convert image data to device
    image = image.float()  # convert image to float precision (cpu)
    return image


def background_sub_frame_helper(frame):
    """Uses the background subtraction method to indentify movement in the video."""
    image = letterbox(frame, stride=64, auto=True)[0] # resizes while maintaining aspect ratio. Resizes to width = 640 pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (27, 27), 0) #blurring an image by a gaussian function to smooth out high frequency noise that could throw out motion detection algorithm off.
    return gray

def yolo_model_helper(background, names, output_data):
    """Runs the YOLO model, plots the outputs to the given image, and returns the processed frame.
    Also calculates the number of detections.
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
                        help='video/0 for webcam')  # video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  # device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  # display results
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')  # box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # boxhideconf
    opt = parser.parse_args()
    return opt


# main function
def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
