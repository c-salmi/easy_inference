import numpy as np
import pyrealsense2 as rs
import math
import cv2
from typing import List

# [x, y, z, w, h, l, theta, class_id, conf]
class BoundingBox3d:
    def __init__(
        self, x, y, z, w, h, l, theta, class_id, confidence, batch_id=0, frame_id=None
    ) -> None:

        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.l = l
        self.theta = theta
        self.class_id = class_id
        self.confidence = confidence

        self.batch_id = batch_id
        self.frame_id = frame_id

    def _roty(self):
        t = self.theta
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @property
    def corners(self):
        R = self._roty()
        l, w, h = self.l, self.w, self.h
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + self.x
        corners_3d[1, :] = corners_3d[1, :] + self.y
        corners_3d[2, :] = corners_3d[2, :] + self.z
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def __str__(self):
        return f"BoundingBox3d(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, w={self.w:.2f}, h={self.h:.2f}, l={self.l:.2f}, class_id={self.class_id}, confidence={self.confidence:.2f}, batch_id={self.batch_id}, frame_id={self.frame_id})"


# [x0, y0, x1, y1, class_id, conf]
class BoundingBox:
    def __init__(
        self, x0, y0, x1, y1, class_id, confidence, batch_id=0, frame_id=None
    ) -> None:

        self._discretize = math.floor

        self.x0 = self._discretize(x0)
        self.x1 = self._discretize(x1)
        self.y0 = self._discretize(y0)
        self.y1 = self._discretize(y1)
        self.class_id = int(class_id)
        self.confidence = confidence
        self.batch_id = int(batch_id)

        # Name of the optical frame of the camera (needed for 3d projection)
        self.frame_id = frame_id

    @property
    def center(self):
        return (int((self.x0 + self.x1) / 2), int((self.y0 + self.y1) / 2))

    @property
    def corners(self):
        return (
            (self.x0, self.y0),
            (self.x1, self.y0),
            (self.x0, self.y1),
            (self.x1, self.y1),
        )

    def to3d(self, depth_frame, intrinsics, depth_scale=0.001) -> BoundingBox3d:

        # TODO: add better heuristics method for depth estimation from bounding box
        # TODO: move the logic of the heuristic to a separate utils function
        depth = (
            np.median(depth_frame[self.y0 : self.y1, self.x0 : self.x1]) * depth_scale
        )

        xyz = rs.rs2_deproject_pixel_to_point(intrinsics, self.center, depth)
        box_points_3d = [
            rs.rs2_deproject_pixel_to_point(intrinsics, corner, depth)
            for corner in self.corners
        ]

        whl = [
            abs(box_points_3d[0][0] - box_points_3d[1][0]),
            abs(box_points_3d[1][1] - box_points_3d[2][1]),
            0.6,
        ]

        return BoundingBox3d(
            *xyz,
            *whl,
            theta=0,
            class_id=self.class_id,
            confidence=self.confidence,
            batch_id=self.batch_id,
        )

    def __str__(self):
        return f"BoundingBox(x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1}, class_id={self.class_id}, confidence={self.confidence:.2f}, batch_id={self.batch_id}, frame_id={self.frame_id})"


def drawBoxes(frame, boxes=List[BoundingBox]):
    if len(frame.shape) == 2:
        color = 255
    else:
        color = (0, 255, 0)

    for box2d in boxes:
        cv2.rectangle(frame, (box2d.x0, box2d.y0), (box2d.x1, box2d.y1), color, 2)
