from ultralytics import YOLO

# proxy 127.0.0.1:7890
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 添加这行

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    results = model.train(data=r".\data.yaml", epochs=200, imgsz=640, batch=48, patience=100, save=True, save_period=50,)
    