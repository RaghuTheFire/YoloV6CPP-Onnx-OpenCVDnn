# YoloV6CPP-Onnx-OpenCVDnn 
YOLOv6 is a single-stage object detection framework dedicated to industrial applications.
YOLOv6 is a recent addition to the long line of state-of-the-art object detection YOLO models, which has shown quite a decent performance with the detection and classification of various objects with greater accuracy and inference speeds.

# Model
Add the model yolov6s.onnx in config_files folder 

# Compilation Command
g++ YoloV6.cpp -o YoloV6 `pkg-config --cflags --libs opencv4`
