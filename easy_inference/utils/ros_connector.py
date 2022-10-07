from easy_inference.utils.boundingbox import BoundingBox3d
import jsk_recognition_msgs.msg as jsk_msgs
from geometry_msgs.msg import Point, Vector3, PoseStamped
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
        if self._fixed_frame is not None:
            msg.header.frame_id = self._fixed_frame
        else:
            msg.header.frame_id = 'map'
        self._publisherBoxes3d.publish(msg)

        # # NOTE: weird zeros behavior
        # if [abs(box_points_3d[1][0] - box_points_3d[0][0]), abs(box_points_3d[2][1] - box_points_3d[1][1])] == [0., 0.]:
        #     continue
        # box3d.header.frame_id = 'lidar' #f'camera{int(pred[0])+1}_link'
        # new_pose = tf_listener.transformPose('lidar', PoseStamped(header=Header(frame_id=f'camera{int(pred[0])+1}_link'), pose=box3d.pose))
        # box3d.pose = new_pose.pose

