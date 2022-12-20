from easy_inference.utils.boundingbox import BoundingBox3d
from easy_inference.utils.skeleton import Skeleton3d
import jsk_recognition_msgs.msg as jsk_msgs
import visualization_msgs.msg as visualization_msgs
from geometry_msgs.msg import Point, Vector3, PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header
import rospy
import tf
from typing import List

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


class RosConnector():
    def __init__(self, name='person_detection', fixed_frame=None):
        rospy.init_node(name)
        self._tf_listener = tf.TransformListener()
        self._publisherBoxes3d = rospy.Publisher('detections3D', jsk_msgs.BoundingBoxArray, queue_size=1) 
        self._publisherSkeleton3d = rospy.Publisher('skeleton3D', visualization_msgs.MarkerArray, queue_size=1) 
        self._publisherPointcloud = rospy.Publisher("pointcloud", PointCloud2, queue_size=10)

        self._fixed_frame = fixed_frame

    def _to_bb_msg(self, box: BoundingBox3d):
        msg = jsk_msgs.BoundingBox()
        msg.pose.position = Point(x=box.x, y=box.y, z=box.z)
        msg.pose.orientation.w = 1
        msg.header.frame_id = f'camera{int(box.batch_id)+1}_color_optical_frame'
        msg.dimensions = Vector3(box.w, box.h, box.l)

        if self._fixed_frame is not None:
            msg.pose = self._tf_listener.transformPose(
                self._fixed_frame, 
                PoseStamped(
                    header=Header(frame_id=f'camera{int(box.batch_id)+1}_color_optical_frame'),
                    pose=msg.pose
                )
            )
            msg.header.frame_id = self._fixed_frame

        return msg

    def publishBoundingBoxes3d(self, boxes: List[BoundingBox3d]):
        msg = jsk_msgs.BoundingBoxArray()
        msg.boxes = [self._to_bb_msg(box) for box in boxes]
        msg.header.stamp = rospy.Time.now()
        if self._fixed_frame == None:
            msg.header.frame_id = f'camera{int(boxes[0].batch_id)+1}_color_optical_frame'
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
                m.id = kpt_id + ((p_id+1)*26)
                m.header.frame_id = f'camera{int(person.batch_id)+1}_color_optical_frame'
                m.type = visualization_msgs.Marker.SPHERE
                m.action = visualization_msgs.Marker.ADD
                m.pose.position = Point(x=x, y=y, z=z)
                m.scale = Point(x=0.05, y=0.05, z=0.05)
                m.pose.orientation.w = 1
                m.lifetime = rospy.Duration(1/20)
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
                m.id = sk_id + ((p_id+1+17)*26)
                m.header.frame_id = f'camera{int(person.batch_id)+1}_color_optical_frame'
                m.type = visualization_msgs.Marker.LINE_STRIP
                m.action = visualization_msgs.Marker.ADD
                m.points = [Point(*kpt0[:3]), Point(*kpt1[:3])]
                m.scale = Point(x=0.02, y=0.0, z=0.0)
                m.lifetime = rospy.Duration(1/20)
                r, g, b = Skeleton3d.LIMB_COLOR[sk_id]
                m.color.r = r
                m.color.g = g
                m.color.b = b
                m.color.a = 1.0
                skeleton_msg.markers.append(m)

        self._publisherSkeleton3d.publish(skeleton_msg)

    def publishPointcloud(self, points, batch_id=0):
        # Create a PointCloud2 message
        msg = PointCloud2()

        # Fill in the fields of the message
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = f'camera{batch_id+1}_color_optical_frame'
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = points.tostring()

        self._publisherPointcloud.publish(msg)

