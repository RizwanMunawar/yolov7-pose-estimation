# yolov7-pose-estimation

## Steps to run Code
- Clone the repository.
```
git clone https://github.com/kunzhi-yu/yolov7-pose.git
```

- Goto the cloned folder.
```
cd yolov7-pose
```

- Create a virtual environment (Recommended, If you dont want to disturb python packages)
```
### For Linux/Mac Users
python3 -m venv yolo-venv
source yolo-venv/bin/activate

### For Window Users
python3 -m venv yolo-venv
cd yolo-venv
cd Scripts
activate
cd ..
cd ..
```
- Upgrade pip.
```
pip install --upgrade pip
```
- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}

### On Jetson Devices

- Install requirements with mentioned command below.

```
pip install -r requirements-jetson.txt
```

- You must install Torch and Torchvision separately. Follow the instructions [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).

### On non-Jetson Devices

- Install requirements with mentioned command below.

```
pip install -r requirements-general.txt
```

## Running the Code

There are a few important arguments you may use.

- `--source` can either be 0/1 for camera, or a video path. No default.
- `--anonymize`add this flag if you wish to anonymize
- `--device` use `cpu` for CPU and `0` for GPU. Default is CPU.
- `--min-area` in background subtraction, the minimal area with changes to be considered motion. Default is 4000.
- `--thresh-val` in background subtraction, the minimal change in pixel value for that pixel to be considered different. Default is 40.
- `--yolo-conf` in the YOLO model, the minimum confidence level for a detection. Default is 0.4.

Two files will be created after the code is executed.
- The output video file: `output_videos/<your-filename-no-extension>_yolo_sub.mp4`
- A CSV file with details saved every 1 second: `output_videos/<your-filename-no-extension>_yolo_sub.csv`

Sample usage
```
# For live, non-anonymize video with default parameters on CPU
python pose-estimate.py --source 0

# For source file and GPU
python pose-estimate.py --source "your custom video.mp4" --device 0

# For live video and custom confidence levels
python pose-estimate.py --source 0 --min-area 1000 --tresh-val 50 --yolo-conf 0.6

# For live, anonymized video
python pose-estimate.py --source 0 --anonymize
