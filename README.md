# Easy Inference

Welcome to the easy inference repository! The main goal of this repository is to provide a clean, simple and short way of setting up inference pipelines for 2D (and 3D) visual detection.
The interfaces to camera drivers are abstracted away as python `generators`. A simple inference pipeline for a webcam based inference pipeline looks as follows:

```Python3
from easy_inference.providers.webcam import Webcam

provider = Webcam(source=0)

for frame in provider:

  # run my detection 
```

See the examples directory for some `yolov7` pipelines.

