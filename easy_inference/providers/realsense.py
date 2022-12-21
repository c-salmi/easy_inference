from easy_inference.providers.provider_base import FrameProvider
import pyrealsense2 as rs
import numpy as np

class Realsense(FrameProvider):
    def __init__(self, width=1280, height=720, depth=False, pointcloud=False, device=None) -> None:
        super().__init__()
        self._pipe = rs.pipeline()
        config = rs.config()
        if device:
            config.enable_device(device)
        self._depth = depth
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        if depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
            self._align = rs.align(rs.stream.color)
        self._profile = self._pipe.start(config)

        self._depth_intr = self._profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._pointcloud = pointcloud
        if pointcloud:
            self._pc = rs.pointcloud()

    def __iter__(self): return self

    def __next__(self):
        self.log_fps()
        frames = self._pipe.wait_for_frames()

        if self._pointcloud and not self._depth:
            raise Exception("It is not possible to enable pointcloud, without depth")

        if not self._depth and not self.pointcloud:
            return np.asanyarray(frames.get_color_frame().get_data())

        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not self._pointcloud:
            return (np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data()))
        else:
            points = self._pc.calculate(depth_frame)
            v = points.get_vertices()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            return (np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data()), verts)

