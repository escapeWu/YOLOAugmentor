from ultralytics import YOLO
model = YOLO("bloodbar.pt")
model.export(format='onnx')