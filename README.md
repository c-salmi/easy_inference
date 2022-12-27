# Easy Inference

![Formatter](https://github.com/c-salmi/easy_inference/actions/workflows/black.yaml/badge.svg)

Welcome to the easy inference repository! The main goal of this repository is to provide a clean, simple and short way of setting up inference pipelines for 2D (and 3D) visual detection.
The interfaces to camera drivers are abstracted away as python `generators`. A simple inference pipeline for a webcam based inference pipeline looks as follows:

```Python3
from easy_inference.providers.webcam import Webcam

provider = Webcam(source=0)

for frame in provider:

  # run my detection 
```

See the examples directory for some `yolov7` pipelines.


### Sidenote

Many examples is this repository include `yolov7` pipelines using the onnx-runtime. To generate onnx model files follow the steps on the yolov7 repository [readme](https://github.com/WongKinYiu/yolov7/blob/8c0bf3f78947a2e81a1d552903b4934777acfa5f/README.md?plain=1#L156).

To export a model for yolov7 including pose detection, checkout the following branch of [yolov7](https://github.com/WongKinYiu/yolov7/tree/pose) and download the pytorch model [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt). Then run the following to export an onnx model with your desired configuration:

```bash
cd yolov7
python models/export.py --weights yolov7-w6-pose.pt --grid --simplify --export-nms --batch-size 2 --img-size 512 640
```

