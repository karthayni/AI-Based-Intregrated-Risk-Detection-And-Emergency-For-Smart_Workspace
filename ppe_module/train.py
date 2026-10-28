"""Step 2: train YOLOv8 on a PPE dataset.
Usage:  python train.py --data data.yaml --epochs 50
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data.yaml")
ap.add_argument("--model", default="yolov8s.pt", help="yolov8n.pt = fastest, yolov8s.pt = good balance")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16, help="lower to 8 or 4 if you run out of memory")
ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
a = ap.parse_args()

model = YOLO(a.model)                      # starts from COCO-pretrained weights (transfer learning)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            device=a.device, project="runs", name="ppe", patience=15)

metrics = model.val()                      # evaluates on the validation split
print(f"mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
print("Best weights saved at: runs/ppe/weights/best.pt")