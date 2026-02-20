from roboflow import Roboflow

rf = Roboflow(api_key="KwQD08ocPgxKEuBZ5t6N")
project = rf.workspace("yolov11modeltraining").project("wild-life-object-detection-ia6sd")
version = project.version(2) # version number from roboflow

# downloads images in YOLO format
dataset = version.download("yolov11", location="./datasets")
