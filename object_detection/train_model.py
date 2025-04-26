# !pip install ultralytics

from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = YOLO("yolo11n.pt")

results = model.train(
    data="/kaggle/input/detection-data/data.yaml",
    epochs=100,
    imgsz=640,
    device=device
)

# !zip -r kaggle-object-detection.zip /kaggle/working/