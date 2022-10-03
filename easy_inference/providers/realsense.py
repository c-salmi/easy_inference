from easy_inference.providers.provider_base import FrameProvider
import pyrealsense2 as rs
import numpy as np

class Realsense(FrameProvider):
    def __init__(self, width=1280, height=720, depth=False, device=None) -> None:
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

    def __next__(self):
        self.log_fps()
        frames = self._pipe.wait_for_frames()

        if self._depth:
            aligned_frames = self._align.process(frames)
            return (np.asanyarray(aligned_frames[0].get_data()), np.asanyarray(aligned_frames[1].get_data()))
        return np.asanyarray(frames[0].get_data())
