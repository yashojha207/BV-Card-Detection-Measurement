from ultralytics import YOLO

model = YOLO("models/card_detector.pt")

results = model.predict(
    source="test_images",
    conf=0.25,
    save=True,
    show=False
)