import pyrealsense2 as rs
import numpy as np
import time
import logger
from typing import Generator
from abc import ABC, abstractmethod

class FrameProvider(ABC):
    def __init__(self) -> None:
        self._last_time = time.monotonic()
        super().__init__()

    def log_fps(self) -> None:
        " Call this method somewhere in the __next__ method to log the fps of the generator loop."
        # TODO: add python logging support
        # print(f'FPS: {1/(time.monotonic() - self._last_time):.2f}')
        self._last_time = time.monotonic() 

    def __iter__(self) -> Generator[int, np.ndarray, str]:
        pass

    @abstractmethod
    def __next__(self) -> np.ndarray:
        pass

