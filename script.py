from ultralytics import YOLO

# Load the exported ONNX model
model = YOLO('Trained_Data_100.pt')

# Run interface on the source
results = model(source=0, show=True, conf=0.40) # generator of results object
