# Configurations when training a model
```bash
imgsz=640
```
For PI 5, we use 640 since if we use anything above, the PI can crash due to it's specs
```bash
epochs=100
```
epochs is the amount that the computer goes over the whole dataset  
batch is based on hardware, more specifically CPU/GPU/RAM (keep small to avoid memory issues)
```bash
batch=8
```
We can put batch=8 but if we are run out of RAM, we have to lower it at least from 4-8  
```bash
device=cpu
```
using cpu due to the fact we are using the PI hardware
```bash
yolo train model=yolo11n.pt data=path/to/data.yaml imgsz=640 batch=8 epochs=100 device=cpu name=new_test
```
example above of how training the model would be with the paramaters used  
To force the camera to stop operating
```bash
ctrl+C
```
The "anything.yaml" represents the dataset configuration file  
For example:
```bash
/home/sunnysquad/ultralytics/datasets/coco8.yaml
```
This means that coco8.yaml is the congifuration file that contains train images, val images, # of classes, class names
