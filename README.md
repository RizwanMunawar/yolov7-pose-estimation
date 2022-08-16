# yolov7-pose-estimation

#### Steps to run Code

- Clone the repository.
```
git clone https://github.com/RizwanMunawar/yolov7-pose-estimation.git
```

- Goto the cloned folder.
```
cd yolov7-pose-estimation
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv psestenv
source psestenv/bin/activate

### For Window Users
python3 -m venv psestenv
cd psestenv
cd Scripts
activate
cd ..
cd ..
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.

```
pip install -r requirements.txt
```

- Download [yolov7-pose-w6.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) file and move it to the working directory {yolov7-pose-estimation}

- Run the code with mentioned command below.
```
python pose-estimate.py

#if you want to change source file
python pose-estimate.py --source "your custom video.mp4"
```

- Output file will be created in the working directory with name ["your-file-name-without-extension"+"__keypoint.mp4"]


https://user-images.githubusercontent.com/62513924/184828185-c8b486ac-656e-4929-95bb-f25404cc4e5b.mp4


#### RESULTS



#### References
- https://github.com/WongKinYiu/yolov7
- https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
- https://github.com/ultralytics/yolov5
