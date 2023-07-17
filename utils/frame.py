import cv2
import numpy as np
import torch
from torchvision import transforms

from utils.datasets import letterbox


class ProcessedFrame:
    """A class representing a processed image frame.

    Attributes:
    processed_frame: An RGB image that is the outcome from processing. Can take multiple image forms depending on CPU or
    GPU choice.
    is_motion: a boolean representing whether there is motion detected in this frame
    num_detections: an integer representing the number of humans detected in the frame
    bed_occupied: a boolean representing whether the bed is occupied.
    """
    def __init__(self, processed_frame, is_motion=False, num_detections=0, bed_occupied=False):
        self.processed_frame = processed_frame
        self.is_motion = is_motion
        self.num_detections = num_detections
        self.bed_occupied = bed_occupied

    def set_is_motion(self, is_motion):
        self.is_motion = is_motion

    def set_num_detections(self, num_detections):
        self.num_detections = num_detections

    def set_bed_occupied(self, bed_occupied):
        self.bed_occupied = bed_occupied

    @property
    def get_processed_frame(self):
        return self.processed_frame

    @property
    def get_is_motion(self):
        return self.is_motion

    @property
    def get_num_detections(self):
        return self.num_detections

    @property
    def get_bed_occupied(self):
        return self.bed_occupied


def background_sub_frame_prep(frame):
    """
    Prepares the frame to be used in background subtraction. The frame is converted to the same size as the YOLO
    model frame and converted to grayscale with blur. The blurring ensures high frequency noise doesn't throw off
    the algorithm.
    """
    image = letterbox(frame, stride=64, auto=True)[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (27, 27), 0)
    return gray


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
