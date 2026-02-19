# Train model with Jetson
Thus we 
Make sure you are within wanted folder so that the model is installed within the desired folder: Thus we "cd" into the ultralytics folder to have yolov11n.pt installed in there.  
Also check the yolo settings to have the runs and datasets pe placed where wanted:
```bash
yolo settings
```

```bash
 yolo train model=yolo11n.pt data=/home/sunnysquad/yolo_project/ultralytics/cfg/datasets/coco8.yaml epochs=10 imgsz=640 batch=2 workers=0 device=0 name=test_camera
```
We can cahnge the batch and workers but system might crash due to not enough ram available.  
Now we test the model:
```bash
yolo predict model=/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_camera/weights/best.pt source=0 show=True
```
