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

if ROS:
    from easy_inference.utils.ros_connector import RosConnector
    ros_connector = RosConnector()

# ort.set_default_logger_severity(0)
ort_sess = ort.InferenceSession('yolov7-w6-pose.onnx', providers=['CUDAExecutionProvider'])

width, height = 640, 480
extra = Realsense(width=width, height=height, depth=True, pointcloud=True, device='043422250695')
providers = [extra]

for frames in combine(*providers):
    rgb_frames = np.stack([f[0] for f in frames]).transpose((0, 3, 1, 2))
    depth_frames = np.stack([f[1] for f in frames])
    pointclouds = np.stack([f[2] for f in frames])

    network_input_dim = [512, 640]
    rgb_frames = pad_frames(rgb_frames, width=network_input_dim[0], height=network_input_dim[1])
    depth_frames = pad_frames(depth_frames, width=network_input_dim[0], height=network_input_dim[1])

    input = np.ascontiguousarray(rgb_frames)
    input = input.astype(np.float32)
    input /= 255

    # output: [batch_id, x0, y0, x1, y1, class_id, conf]
    outputs = ort_sess.run(None, {'images': input})[0]

    # convert to Skeleton for convenience
    persons = [Skeleton(*output[0:4], class_id=output[5], confidence=output[4], kpts=output[6:], batch_id=0) for output in outputs]

    # filter classes
    persons = [person for person in persons if person.class_id == 0]

    # filter confidence
    persons = [person for person in persons if person.confidence > 0.8]

    persons3d = [person.to3d(depth_frames[person.batch_id], providers[person.batch_id]._depth_intr) for person in persons]

    if ROS and len(persons3d)>0: 
        ros_connector.publishPersons3d(persons3d)
        batch_id = 0
        ros_connector.publishPointcloud(pointclouds[batch_id], batch_id)
    
    if SHOW:
        f_rgb = np.ascontiguousarray(rgb_frames[0].transpose(1,2,0).astype(np.uint8))
        f_depth = depth_frames[0] * (255/np.amax(depth_frames[0]))

        drawBoxes(f_rgb, persons)
        drawBoxes(f_depth, persons)

        drawSkeletons(f_rgb, persons)
        drawSkeletons(f_depth, persons)

        cv2.imshow('depth_frame', f_depth)
        cv2.imshow('rgb_frame', f_rgb)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()

