from ultralytics import YOLO

def main():
    # 1. Load the YOLOv11 Nano model
    model = YOLO("yolo11n.pt")

    # 2. Start the training
    model.train(
        data="./datasets/data.yaml",    # Path to the file Roboflow just downloaded
        epochs=100,                     # Number of training rounds
        imgsz=640,                      # Image size
        batch=64,                       # Batch size (Higher Batch size = greater gpu usage)
        project="Wildlife_Project",     # Saves results in this folder
        name="Boar_Deer_Squirrel",      # Specific run name
        device=0,                       # Which GPU will be used (Make sure GPU is being used and not on board graphics)
        workers=8                       # Number of CPU cores to help load images
    )

if __name__ == "__main__":
    main()
