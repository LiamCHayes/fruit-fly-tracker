"""Testing the fine-tined model"""

from ultralytics import YOLO

model = YOLO("runs/train/weights/best.pt")

# Inference on synthetic data
results = model("../data_generation/dataset/yolo_dataset/images/val/counter_with_pen-00003-06.jpeg")
for r in results:
    print(r.boxes)
    r.show()

# Inference on real data
results = model("../data_generation/dataset/yolo_dataset/images/val/counter_with_pen-00003-06.jpeg")
for r in results:
    print(r.boxes)
    r.show()
