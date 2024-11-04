from ultralytics import YOLO


model = YOLO('yolov8n.pt')

model.train(
    data='./yolo.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device="mps"
)

results = model.predict('./datasets/ladolt_dataset/images/val/0.png')
results.show()
