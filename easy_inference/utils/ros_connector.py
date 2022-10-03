from easy_inference.utils.boundingbox import BoundingBox3d
import jsk_recognition_msgs.msg as jsk_msgs
from geometry_msgs.msg import Point, Vector3
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
    def __init__(self, name='person_detection'):
        rospy.init_node(name)
        self._tf_listener = tf.TransformListener()
        # publisher = rospy.Publisher('/detections3D', BoundingBox3DArray, queue_size=1) 
        self._publisherBoxes3d = rospy.Publisher('detections3D', jsk_msgs.BoundingBoxArray, queue_size=1) 

    def _to_bb_msg(self, box: BoundingBox3d):
        msg = jsk_msgs.BoundingBox()
        msg.pose.position = Point(x=box.x, y=box.y, z=box.z)
        msg.pose.orientation.w = 1
        msg.dimensions = Vector3(box.w, box.h, box.l)
        msg.header.frame_id = f'camera{int(box.batch_id)+1}_color_optical_frame'
        return msg

    def publishBoundingBoxes3d(self, boxes: List[BoundingBox3d]):
        msg = jsk_msgs.BoundingBoxArray()
        msg.boxes = [self._to_bb_msg(box) for box in boxes]
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'velodyne'
        self._publisherBoxes3d.publish(msg)

        # # NOTE: weird zeros behavior
        # if [abs(box_points_3d[1][0] - box_points_3d[0][0]), abs(box_points_3d[2][1] - box_points_3d[1][1])] == [0., 0.]:
        #     continue
        # box3d.header.frame_id = 'lidar' #f'camera{int(pred[0])+1}_link'
        # new_pose = tf_listener.transformPose('lidar', PoseStamped(header=Header(frame_id=f'camera{int(pred[0])+1}_link'), pose=box3d.pose))
        # box3d.pose = new_pose.pose

