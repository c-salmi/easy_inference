from providers import Webcam, combine
# from processors import filter_iou3d, realsense_to3d
from utils import BoundingBox, BoundingBox3d

import onnxruntime as ort
import numpy as np
import pyrealsense2 as rs
import cv2
import time 

SHOW = True
ROS = True

# if ROS:
#     from ros_connector import RosConnector
#     ros_connector = RosConnector()

# ort.set_default_logger_severity(0)
ort_sess = ort.InferenceSession('yolov7-tiny.onnx', providers=['CUDAExecutionProvider'])

webcam = Webcam(source=0)
providers = [webcam]

for frames in combine(*providers):
    rgb_frames = np.stack([f[1] for f in frames])
    depth_frames = np.stack([f[0] for f in frames])

    input = rgb_frames.transpose((0, 3, 1, 2))
    input = np.ascontiguousarray(input)
    input = input.astype(np.float32)
    input /= 255

    # output: [batch_id, x0, y0, x1, y1, class_id, conf]
    outputs = ort_sess.run(None, {'images': input})[0]

    # convert to BoundingBox for convenience
    boxes2d = [BoundingBox(*output[1:], batch_id=output[0]) for output in outputs]

    # filter classes
    # boxes2d = boxes2d[boxes2d[:, -2] != 0.]

    # project to BoundingBox3d
    boxes3d = [box2d.to3d(depth_frame=depth_frames[box2d.batch_id], instrinsics=providers[box2d.batch_id]._depth_intr) for box2d in boxes2d]

    # filter iou overlap
    # boxes3d = filter_iou3d(boxes3d)    
    #
    # if ROS: 
    #     ros_connector.publishBoxes3d(boxes3d)
    #
    # if SHOW:
    #     for box2d in boxes2d:
    #         #only person class
    #         if box2d[-2] != 0.: continue
    #
    #         box = box2d[1:5]
    #         box = box.round().astype(np.int32).tolist()
    #         cv2.rectangle(rgb_frames[box2d[0]], box[:2], box[2:],(0,255,0),2)
    #
    #     cv2.imshow(f'frame', np.concatenate(
    #         (np.concatenate(rgb_frames[:3], axis=1), 
    #         np.concatenate((rgb_frames[3], rgb_frames[4], np.zeros_like(rgb_frames[4])), axis=1)),
    #         axis=0))
    #
    #     if cv2.waitKey(1) == 27:
    #         break

cv2.destroyAllWindows()

