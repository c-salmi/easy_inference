from easy_inference.providers.realsense import Realsense
from easy_inference.providers.utils import combine
from easy_inference.utils.boundingbox import BoundingBox, drawBoxes
from easy_inference.utils.filters import filter_iou3d

import onnxruntime as ort
import numpy as np
import pyrealsense2 as rs
import cv2
import os

ROS = os.getenv("ROS", 0)
SHOW = os.getenv("SHOW", 0)

if ROS:
    from easy_inference.utils.ros_connector import RosConnector

    ros_connector = RosConnector()

# ort.set_default_logger_severity(0)
ort_sess = ort.InferenceSession("yolov7-tiny.onnx", providers=["CUDAExecutionProvider"])

extra = Realsense(width=640, height=480, depth=True, device="043422250695")
providers = [extra]

for frames in combine(*providers):
    rgb_frames = np.stack([f[0] for f in frames])
    depth_frames = np.stack([f[1] for f in frames])

    input = rgb_frames.transpose((0, 3, 1, 2))
    input = np.ascontiguousarray(input)
    input = input.astype(np.float32)
    input /= 255

    # output: [batch_id, x0, y0, x1, y1, class_id, conf]
    outputs = ort_sess.run(None, {"images": input})[0]

    # convert to BoundingBox for convenience
    boxes2d = [BoundingBox(*output[1:], batch_id=output[0]) for output in outputs]

    # filter classes
    boxes2d = [box2d for box2d in boxes2d if box2d.class_id == 0]

    # project to BoundingBox3d
    boxes3d = [
        box2d.to3d(depth_frames[box2d.batch_id], providers[box2d.batch_id]._depth_intr)
        for box2d in boxes2d
    ]

    # filter iou overlap
    boxes3d = filter_iou3d(boxes3d)

    if ROS:
        ros_connector.publishBoundingBoxes3d(boxes3d)
    #
    if SHOW:
        f = rgb_frames[0]
        drawBoxes(frame=f, boxes=boxes2d)

        cv2.imshow("frame", f)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
