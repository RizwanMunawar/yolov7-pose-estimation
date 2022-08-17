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

- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}


- Run the code with mentioned command below.
```
python pose-estimate.py

#if you want to change source file
python pose-estimate.py --source "your custom video.mp4"

#For CPU
python pose-estimate.py --source "your custom video.mp4" --device cpu

#For GPU
python pose-estimate.py --source "your custom video.mp4" --device 0

```

- Output file will be created in the working directory with name <b>["your-file-name-without-extension"+"_keypoint.mp4"]</b>

#### RESULTS

<center><p>
  <img src="https://user-images.githubusercontent.com/62513924/185089411-3f9ae391-ec23-4ca2-aba0-abf3c9991050.png" width="416" />
  <img src="https://user-images.githubusercontent.com/62513924/185228806-4ba62e7a-12ef-4965-a44a-6b5ba9a3bf28.png" width="450" /> 
</p></center>


#### References
- https://github.com/WongKinYiu/yolov7
- https://github.com/augmentedstartups/yolov7
- https://github.com/augmentedstartups
- https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
- https://github.com/ultralytics/yolov5

#### My Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
