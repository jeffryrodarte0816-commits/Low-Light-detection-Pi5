# To set up and configure the Arducam(day/night camera)
We can dowload a software called qv412 but we get it to wehre it deletes other files, unless we nstall wihtout deleting anyhting
To test at the moment with what we have in the RPI5, we can use the yolo predict:  
```bash
yolo predict model=/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_regcamera2/weights/best.pt source=0 show=True  //the source=0 represents for example the camera
```

