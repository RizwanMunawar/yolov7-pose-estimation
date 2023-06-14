import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt


@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="preped_videos/7855_test.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
    frame_count = 0  # count no of frames
    total_fps = 0  # count total fps

    device = select_device(opt.device)  # select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  # Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

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

        first_frame_init = letterbox(cap.read()[1], stride=64, auto=True)[0]  # init videowriter
        resize_height, resize_width = first_frame_init.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_yolo.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (resize_width, resize_height))

        # create an empty frame to overlay
        blank_frame = np.zeros((resize_height, resize_width, 3), dtype=np.uint8)
        blank_frame = cv2.cvtColor(blank_frame, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)

        while cap.isOpened:  # loop until cap opened or video not complete

            print("Frame {} Processing".format(frame_count + 1))

            ret, frame = cap.read()  # get frame and success from video capture

            background = first_frame_init.copy()
            # background = blank_frame.copy()

            if ret:  # if success is true, means frame exist
                orig_image = frame  # store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
                image = letterbox(image, stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)  # convert image data to device
                image = image.float()  # convert image to float precision (cpu)
                start_time = time.time()  # start time for fps calculation

                with torch.no_grad():  # get predictions
                    output_data, _ = model(image)

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

                for i, pose in enumerate(output_data):  # detections per image
                    if len(output_data) and len(pose[:, 5].unique()) != 0:  # check if no pose
                        for c in pose[:, 5].unique():  # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            # print("No of Objects in Current Frame : {}".format(n))
                            cv2.putText(background, "No. detections: {}".format(n), (10, 20),
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
                        print("empty")
                        cv2.putText(background, "Room empty", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1)

                end_time = time.time()  # Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                # Stream results
                # if view_img:
                #     cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                #     cv2.waitKey(1)  # 1 millisecond

                out.write(background)  # writing the video frame

            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


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
