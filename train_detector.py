from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="detector_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    workers=0,
    device="cpu",
    project="runs",
    name="card_detector"
)