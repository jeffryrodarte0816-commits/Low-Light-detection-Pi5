# Making a model for Object detection
1. make new model since we need a best.pt and last.pt which the "train" model in file did not have
2. once we make new model, make sure it has the best.pt and last.pt so that we can implement the camera to test the actual model itself
3. now to make new model it is
```bash
yolo train model=yolo11n.pt data=/home/sunnysquad/ultralytics/datasets/coco8.yaml epochs=100 lr0=0.01 name=my_new_model
```
4. what the above line did was change the name of the model created to be called "my_newmodel"
5. inside the files we have now within the detection aspect of Yolo, the model named my_new_model
```bash
 /home/sunnysquad/ultralytics/runs/detect/my_new_model/    /// this is the directory shown in terminal
```
6. we use as recommended the best.pt which is in general better than last.pt to test with camera examble being using the model named "train"
```bash
yolo predict model=/home/sunnysquad/ultralytics/runs/detect/my_new_model/weights/best.pt source=0   //the source=0 represents for example the camera
```

# Configuring camera connection to YOLO
look into what source would be for whatever camera we use for project and make sure it's set to be equal to source with above code  
## In the case of 1080 Pro stream
What it means below is that 1080P camera  
```bash
1080P Pro Stream (usb-xhci-hcd.1-1):
	/dev/video0
	/dev/video1
	/dev/media3
```
video0	is the Main video stream (what YOLO should use)  
video1	is Secondary stream (IR, depth, control channel, etc., depending on camera)  
media3	 is the Media controller interface (not used for YOLO)
Under is how to play live feed of 1080 P knowing that video0 is main video stream
```bash
ffplay /dev/video0
```
If camera shows up with video0 then try running the model on the video0
```bash
yolo detect predict model=/home/sunnysquad/ultralytics/runs/detect/test_camera/weights/best.pt source=0 show=True
```
What this should do is show the camera along with objects w/label and show in terminal the # of objects w/label as well  
# To Continue training a model
1.Add training images along with it's corresponding txt file  
2.Continue to train the model you are working with for example using the model "test_camera" we do  
```bash
yolo train \
model=/home/sunnysquad/ultralytics/runs/detect/test_camera/weights/best.pt \
data=/home/sunnysquad/ultralytics/datasets/coco8.yaml \
epochs=50 \
imgsz=640
```
What this does is runs the "test_camera" again, updates the best.pt and last.pt from training the model  
Lastly, to retest the model do:
```bash
yolo detect predict model=/home/sunnysquad/ultralytics/runs/detect/test_camera/weights/best.pt source=0 show=True
```
This should show like explained before the model operating w/camera
# To see better results, do this for training
1. collect new data
2.  fixed the wrong labels
3.   add new classes, if only humans then add like dog/cat
4. fine tune the best.pt,meaning using best.pt as starting point  
For example:
```bash
yolo train \
model=/home/sunnysquad/ultralytics/runs/detect/test_camera/weights/best.pt \
data=/home/sunnysquad/ultralytics/datasets/coco8.yaml \
epochs=50 \
imgsz=640
```
# Making model for Instance segmentation

