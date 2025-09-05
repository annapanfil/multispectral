# This script is not ment tu be run on jetson nano 
# matplotlib                3.7.5
# networkx                  3.1
# numpy                     1.24.4
# nvidia-cublas-cu12        12.1.3.1
# nvidia-cuda-cupti-cu12    12.1.105
# nvidia-cuda-nvrtc-cu12    12.1.105
# nvidia-cuda-runtime-cu12  12.1.105
# nvidia-cudnn-cu12         9.1.0.70
# nvidia-cufft-cu12         11.0.2.54
# nvidia-curand-cu12        10.3.2.106
# nvidia-cusolver-cu12      11.4.5.107
# nvidia-cusparse-cu12      12.1.0.106
# nvidia-nccl-cu12          2.20.5
# nvidia-nvjitlink-cu12     12.9.86
# nvidia-nvtx-cu12          12.1.105
# onnx                      1.17.0
# onnxruntime               1.19.2
# onnxslim                  0.1.57
# opencv-python             4.11.0.86
# torch                     2.4.1
# torchvision               0.19.1
# ultralytics               8.3.153
# ultralytics-thop          2.0.14
# urllib3                   2.2.3


from ultralytics import YOLO

model = YOLO('model.pt')

model.export(format='onnx', opset=11, simplify=True, imgsz=(800, 608))

