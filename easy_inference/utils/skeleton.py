import numpy as np
import pyrealsense2 as rs
from  easy_inference.utils.boundingbox import BoundingBox, BoundingBox3d
import math
import cv2
from typing import List

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

limbs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


class Skeleton3d(BoundingBox3d):
    LIMBS=limbs
    LIMB_COLOR=limb_color
    KPT_COLOR=kpt_color

    @property
    def keypoints(self):
        return self.kpts

# [x0, y0, x1, y1, class_id, conf, [x_kpt, y_ktp, conf_kpt] ]
class Skeleton(BoundingBox):
    LIMBS=limbs
    LIMB_COLOR=limb_color
    KPT_COLOR=kpt_color

    def __init__(self, x0, y0, x1, y1, class_id, confidence, kpts, batch_id=0, frame_id=None) -> None:
        super().__init__(x0, y0, x1, y1, class_id, confidence, batch_id, frame_id)

        # kpts
        self.kpts = [(self._discretize(kpts[3*i]), self._discretize(kpts[3*i+1]), kpts[3*i+2]) for i in range(len(kpts)//3)]
        assert len(self.kpts) == 17

    @property
    def keypoints(self):
        return self.kpts

    def to3d(self, depth_frame, intrinsics, depth_scale=0.001) -> Skeleton3d:
        box3d = super().to3d(depth_frame, intrinsics, depth_scale)

        assert len(self.keypoints) == 17

        kpts = []
        depth_estimation_margin = 2
        for kpt_id, kpt in enumerate(self.keypoints):
            patch = depth_frame[
                kpt[1]-depth_estimation_margin: kpt[1]+depth_estimation_margin,
                kpt[0]-depth_estimation_margin: kpt[0]+depth_estimation_margin
            ].flatten()
            patch = patch[np.nonzero(patch)]
            depth_kpt = np.median(patch)*depth_scale

            if np.isnan(depth_kpt):
                xyz = [-1, -1, -1]
                conf = 0.0
            else:
                xyz = rs.rs2_deproject_pixel_to_point(intrinsics, kpt[:2], depth_kpt)
                conf = kpt[2]

            kpts.append((*xyz, conf, kpt_id))

        box3d.__class__ = Skeleton3d
        box3d.kpts = kpts

        return box3d

    def __str__(self):
        return f'BoundingBox(x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1}, class_id={self.class_id}, confidence={self.confidence:.2f}, batch_id={self.batch_id}, frame_id={self.frame_id})'


def drawSkeletons(frame, skeletons: List[Skeleton], radius=5):
    for skeleton in skeletons:
        num_kpts = len(skeleton.keypoints)

        for i, (x, y, conf) in enumerate(skeleton.keypoints):
            if conf < 0.5: continue

            if len(frame.shape) == 2:
                color = 255
            else:
                color = tuple([int(c) for c in Skeleton.KPT_COLOR[i]])

            cv2.circle(frame, (x, y), radius, color, -1)

        for sk_id, sk in enumerate(Skeleton.LIMBS):
            kpt0 = skeleton.keypoints[sk[0]-1]
            kpt1 = skeleton.keypoints[sk[1]-1]

            if len(frame.shape) == 2:
                color = 255
            else:
                color = tuple([int(c) for c in Skeleton.LIMB_COLOR[sk_id]])

            if kpt0[2]>0.5 and kpt1[1]>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
                cv2.line(frame, kpt0[:2], kpt1[:2], color, thickness=2)


