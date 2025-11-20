# Tran model with Jetson
Make sure you are within wanted folder so that the model is installed within the desired folder
```bash
 yolo train model=yolo11n.pt data=/home/sunnysquad/yolo_project/ultralytics/cfg/datasets/coco8.yaml epochs=10 imgsz=640 batch=2 workers=0 device=0 name=test_camera
```
This model thats trained does not go into ultralytics since it was installed in home folder  
Now we test the model:
```bash
yolo predict model=/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_camera/weights/best.pt source=0 show=True
```
Now what i need to do is reconfigure the files that the model runs and test on, and look into rpi5 to see how the things were reconfigured.
