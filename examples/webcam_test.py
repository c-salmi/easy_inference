from easy_inference.providers.webcam import Webcam
from easy_inference.providers.utils import combine
from easy_inference.utils.boundingbox import BoundingBox, drawBoxes

import os
import onnxruntime as ort
import numpy as np
import pyrealsense2 as rs
import cv2

ROS = os.getenv("ROS", 0)
SHOW = os.getenv("SHOW", 0)

# ort.set_default_logger_severity(0)
ort_sess = ort.InferenceSession('yolov7-tiny.onnx', providers=['CUDAExecutionProvider'])

webcam = Webcam(source=0)

for rgb_frame in webcam:
    input = rgb_frame.transpose((2, 0, 1))
    input = np.ascontiguousarray(input)
    input = input.astype(np.float32)
    input /= 255
    input = np.expand_dims(input, 0)

    # output: [batch_id, x0, y0, x1, y1, class_id, conf]
    outputs = ort_sess.run(None, {'images': input})[0]

    # convert to BoundingBox for convenience
    boxes = [BoundingBox(*output[1:]) for output in outputs]

    if SHOW:
        drawBoxes(rgb_frame, boxes)

        cv2.imshow('rgb_frame', rgb_frame)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()

