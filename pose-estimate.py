import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
import argparse
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run(
        poseweights='yolov7-w6-pose.pt',
        source='football1.mp4'):


    #load weights
    weights = torch.load(poseweights)
    model = weights['model']
    model = model.half().to(device)
    _ = model.eval()

    #video path
    video_path = source

    #pass video to videocapture object
    cap = cv2.VideoCapture(video_path)

    #check if videocapture not opened
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    #get video frame width
    frame_width = int(cap.get(3))

    #get video frame height
    frame_height = int(cap.get(4))

    #code to write a video
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{out_video_name}_keypoint.mp4",
                        cv2.VideoWriter_fourcc(*'mp4v'), 30,
                        (resize_width, resize_height))

    #count no of frames
    frame_count = 0
    #count total fps
    total_fps = 0 

    #loop until cap opened or video not complete
    while(cap.isOpened):
        
        print("Frame {} Processing".format(frame_count))
        #get frame and success from video capture
        ret, frame = cap.read()

        #if success is true, means frame exist
        if ret:

            #store frame
            orig_image = frame
            
            #convert frame to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.half()
            start_time = time.time()
            with torch.no_grad():
                output, _ = model(image)

            #Apply non max suppression
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
            im0 = image[0].permute(1, 2, 0) * 255
            im0 = im0.cpu().numpy().astype(np.uint8)
            
            #reshape image format to (BGR)
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            for idx in range(output.shape[0]):
                plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
                xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
                xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
                
                #Plotting key points on Image
                cv2.rectangle(im0,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0),
                    thickness=1,lineType=cv2.LINE_AA)
            
            #Calculatio for FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            cv2.putText(im0, f'FPS: {int(fps)}', (11, 100), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            
            # cv2.imshow('image', nimg)
            out.write(im0)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    # cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='Video/0 for webcam')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
