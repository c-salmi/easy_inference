from easy_inference.providers.realsense import Realsense
from easy_inference.providers.utils import combine
from easy_inference.utils.boundingbox import BoundingBox, drawBoxes
from easy_inference.utils.skeleton import Skeleton, drawSkeletons
from easy_inference.utils.filters import filter_iou3d
from easy_inference.utils.pad_frames import pad_frames

import os
import onnxruntime as ort
import numpy as np
import pyrealsense2 as rs
import cv2

ROS = os.getenv("ROS", 0)
SHOW = os.getenv("SHOW", 0)

# ort.set_default_logger_severity(0)
ort_sess = ort.InferenceSession(
    "yolov7-w6-pose.onnx", providers=["CUDAExecutionProvider"]
)

width, height = 640, 480
# cam1 = Realsense(width=width, height=height, depth=True, pointcloud=True, device='215122255869')
cam1 = Realsense(
    width=width, height=height, depth=True, pointcloud=True, device="043422250695"
)

if ROS:
    from easy_inference.utils.ros_connector import RosConnector

    ros_connector = RosConnector(num_cameras=1)

for (rgb_frame, depth_frame, pointcloud) in cam1:
    rgb_frame = rgb_frame.transpose(2, 0, 1)

    network_input_dim = [640, 512]  # width, height
    rgb_frame = pad_frames(
        rgb_frame, width=network_input_dim[0], height=network_input_dim[1], type="rgb"
    )
    depth_frame = pad_frames(
        depth_frame,
        width=network_input_dim[0],
        height=network_input_dim[1],
        type="depth",
    )

    rgb_frame = rgb_frame.astype(np.float32) / 255

    outputs = ort_sess.run(None, {"images": np.expand_dims(rgb_frame, 0)})[0]

    # convert to Skeleton for convenience
    persons = [
        Skeleton(
            *output[0:4], class_id=output[5], confidence=output[4], kpts=output[6:]
        )
        for output in outputs
    ]

    # filter classes
    persons = [person for person in persons if person.class_id == 0]

    # filter confidence
    persons = [person for person in persons if person.confidence > 0.8]

    # convert to 3d using depth and intrinsics
    persons3d = [person.to3d(depth_frame, cam1._depth_intr) for person in persons]

    if ROS and len(persons3d) > 0:
        ros_connector.publishPersons3d(persons3d)
        ros_connector.publishPointclouds([pointcloud])

    if SHOW:
        f_rgb = np.ascontiguousarray(rgb_frame.transpose(1, 2, 0))
        f_depth = depth_frame * (255 / np.amax(depth_frame))

        drawBoxes(f_rgb, persons)
        drawBoxes(f_depth, persons)

        drawSkeletons(f_rgb, persons)
        drawSkeletons(f_depth, persons)

        cv2.imshow("depth_frame", f_depth)
        cv2.imshow("rgb_frame", f_rgb)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
