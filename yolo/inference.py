"""Testing the fine-tined model"""

from ultralytics import YOLO

model = YOLO("runs/train/weights/best.pt")

# Inference on synthetic data
results = model("../data_generation/dataset/yolo_dataset/images/val/counter_with_pen-00002-01.jpeg")
for r in results:
    print(r.boxes)
    r.show()

# Inference on real data
results = model("../real_data/fruit_fly_cabinet_frames/fruit_fly_cabinet_0001.jpeg")
for r in results:
    print(r.boxes)
    r.show()
