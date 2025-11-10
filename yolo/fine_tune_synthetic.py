"""Fine tunes a yolo detection model based on synthetic data"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
        data="../data_generation/dataset/yolo_dataset/data.yaml",
        project="./runs",
        epochs=100,
        imgsz=1280,
        batch=2)
