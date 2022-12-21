from easy_inference.utils.boundingbox import BoundingBox3d
from easy_inference.utils.skeleton import Skeleton3d
import jsk_recognition_msgs.msg as jsk_msgs
import visualization_msgs.msg as visualization_msgs
from geometry_msgs.msg import Point, Vector3, PoseStamped, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header
import rospy
import tf
import tf2_ros
from typing import List
import time

def transform(translation, rotation):
    trans_x, trans_y, trans_z = translation
    x, y, z, w = rotation
    return [
        [1-2*y**2-2*z**2,    2*x*y-2*w*z,      2*x*z+2*w*y,     trans_x],
        [2*x*y+2*w*z,        1-2*x**2-2*z**2,  2*y*z-2*w*x,     trans_y],
        [2*x*z-2*w*y,        2*y*z+2*w*x,      1-2*x**2-2*y**2, trans_z],
        [0,              0,            0,                 1]
    ]

# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
Point.__iter__ = _it

# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
Vector3.__iter__ = _it

# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
    yield self.w
Quaternion.__iter__ = _it


class RosConnector():
    def __init__(self, name='person_detection', num_cameras=1, fixed_frame=None):
        rospy.init_node(name)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener1 = tf2_ros.TransformListener(self._tf_buffer)
        self._tf_listener = tf.TransformListener()
        time.sleep(1)

        # Convention to name camera frame
        self._local_frames = [f'camera{i+1}_color_optical_frame' for i in range(num_cameras)]
        if fixed_frame is None:
            self._fixed_frame = self._local_frames[0]
        else:
            self._fixed_frame = fixed_frame

        # Publishers
        self._publisherBoxes3d = rospy.Publisher('~detections3D', jsk_msgs.BoundingBoxArray, queue_size=1) 
        self._publisherSkeleton3d = rospy.Publisher('~skeleton3D', visualization_msgs.MarkerArray, queue_size=1) 
        self._publisherPointcloud = rospy.Publisher('~pointcloud', PointCloud2, queue_size=10)

    def _to_bb_msg(self, box: BoundingBox3d):
        msg = jsk_msgs.BoundingBox()
        msg.pose.position = Point(x=box.x, y=box.y, z=box.z)
        msg.pose.orientation.w = 1
        msg.header.frame_id = self._local_frames[box.batch_id]
        msg.dimensions = Vector3(box.w, box.h, box.l)

        if self._fixed_frame is not None:
            msg.pose = self._tf_listener.transformPose(
                self._fixed_frame, 
                PoseStamped(
                    header=msg.header,
                    pose=msg.pose
                )
            ).pose
            msg.header.frame_id = self._fixed_frame

        return msg

    def publishBoundingBoxes3d(self, boxes: List[BoundingBox3d]):
        msg = jsk_msgs.BoundingBoxArray()
        msg.boxes = [self._to_bb_msg(box) for box in boxes]
        msg.header.stamp = rospy.Time.now()
        if self._fixed_frame == None:
            msg.header.frame_id = self._local_frames[boxes[0].batch_id]
        else:
            msg.header.frame_id = self._fixed_frame
        self._publisherBoxes3d.publish(msg)

        # # NOTE: weird zeros behavior
        # if [abs(box_points_3d[1][0] - box_points_3d[0][0]), abs(box_points_3d[2][1] - box_points_3d[1][1])] == [0., 0.]:
        #     continue
        # box3d.header.frame_id = 'lidar' #f'camera{int(pred[0])+1}_link'
        # new_pose = tf_listener.transformPose('lidar', PoseStamped(header=Header(frame_id=f'camera{int(pred[0])+1}_link'), pose=box3d.pose))
        # box3d.pose = new_pose.pose

    def publishPersons3d(self, persons: List[Skeleton3d], conf_threshold=0.5):
        self.publishBoundingBoxes3d(persons)

        skeleton_msg = visualization_msgs.MarkerArray()
        for p_id, person in enumerate(persons):
            for x, y, z, conf, kpt_id in person.keypoints:
                if conf < conf_threshold: continue

                m = visualization_msgs.Marker()
                m.id = kpt_id + (p_id*26)
                m.header.frame_id = self._local_frames[person.batch_id]
                m.type = visualization_msgs.Marker.SPHERE
                m.action = visualization_msgs.Marker.ADD
                m.pose.position = Point(x=x, y=y, z=z)
                m.scale = Point(x=0.05, y=0.05, z=0.05)
                m.pose.orientation.w = 1
                m.lifetime = rospy.Duration(1/10)
                r, g, b = Skeleton3d.KPT_COLOR[kpt_id]
                m.color.r = r
                m.color.g = g
                m.color.b = b
                m.color.a = 1.0
                skeleton_msg.markers.append(m)

            for sk_id, sk in enumerate(Skeleton3d.LIMBS):
                kpt0 = person.keypoints[sk[0]-1]
                kpt1 = person.keypoints[sk[1]-1]
                
                # check confidences
                if kpt0[3]<conf_threshold or kpt1[3]<conf_threshold: 
                    continue

                m = visualization_msgs.Marker()
                m.id = sk_id + (p_id*26) + 17
                m.header.frame_id = self._local_frames[person.batch_id]
                m.type = visualization_msgs.Marker.LINE_STRIP
                m.action = visualization_msgs.Marker.ADD
                m.points = [Point(*kpt0[:3]), Point(*kpt1[:3])]
                m.scale = Point(x=0.02, y=0.0, z=0.0)
                m.lifetime = rospy.Duration(1/10)
                r, g, b = Skeleton3d.LIMB_COLOR[sk_id]
                m.color.r = r
                m.color.g = g
                m.color.b = b
                m.color.a = 1.0
                skeleton_msg.markers.append(m)

        self._publisherSkeleton3d.publish(skeleton_msg)

    def publishPointclouds(self, clouds):
        # TODO: improve speed of pointcloud transformations

        # Create a PointCloud2 message
        msg = PointCloud2()

        all_points = None
        for frame, cloud in zip(self._local_frames, clouds):
            trans = self._tf_buffer.lookup_transform(self._fixed_frame, frame, rospy.Time(0))

            matrix = transform(list(trans.transform.translation), list(trans.transform.rotation))

            cloud = np.concatenate((cloud, np.ones((len(cloud), 1))), axis=1)

            points = (matrix @ cloud.T).T

            if all_points is None:
                all_points = points
            else:
                all_points =  np.concatenate((all_points, points), axis=0)

        # Fill in the fields of the message
        all_points = all_points[:, :3].astype(np.float32)
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._fixed_frame
        msg.height = 1
        msg.width = len(all_points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = all_points.tostring()

        self._publisherPointcloud.publish(msg)

