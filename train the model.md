1. delete the model we have called "train" within ultralytics/runs/detect
2. make new model since we need a best.pt and last.pt which the "train" model in file did not have
3. once we make new model, make sure it has the best.pt and last.pt so that we can implement the camera to test the actual model itself
4. we use as recommended the best.pt which is in general better than last.pt to test with camera examble being using the model named "train"
```bash
yolo predict model=/home/sunnysquad/ultralytics/runs/detect/train/weights/best.pt source=0   //the source=0 represents for example the camera
```
6. look into what source would be for whatever camera we use for project and make sure it's set to be equal to source with above code
7. now to make new model it is
```bash
yolo train model=yolo11n.pt data=/home/sunnysquad/ultralytics/datasets/coco8.yaml epochs=100 lr0=0.01 name=my_new_model
```
8. what the above line did was change the name of the model created to be called "my_newmodel"
9. inside the files we have now within the detection aspect of Yolo, the model named my_new_model
```bash
 /home/sunnysquad/ultralytics/runs/detect/my_new_model/    /// this is the directory shown in terminal
```
