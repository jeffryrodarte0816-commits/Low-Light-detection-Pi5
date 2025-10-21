from ultralytics import YOLO
import kagglehub
import os 
import times


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
bat_results = model.train(data="C:\Users\ajriv\Documents\YOLOv11\archive\animals\animals\bat", epochs=100, imgsz=640)

# prints out statement of models train

def Train_Animals_Results(bat_results):
    print(bat_results)
    for i, results in enumerate(bat_results):
        print(f" {i}: {bat_results}")
    f = open("C:\Users\ajriv\Documents\YOLOv11\archive\animals\animals\bat")
    print(f.read("This animal is ")), f.read(bat_results)

    return Train_Animals_Results(bat_results)

     





