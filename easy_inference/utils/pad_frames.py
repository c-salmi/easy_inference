import numpy as np


def pad_frames(frames, width, height, type="rgb"):
    row_pad, col_pad = ([height, width] - np.array(frames.shape)[-2:]) // 2
    assert row_pad >= 0 and col_pad >= 0

    if len(frames.shape) > 4:
        raise Exception(
            "The function pad_frames doesn't accept a frame array of dimension bigger than 4"
        )
    elif len(frames.shape) < 2:
        raise Exception(
            "The function pad_frames doesn't accept a frame array of dimension smaller than 2"
        )
    elif len(frames.shape) == 4:
        return np.pad(frames, ((0, 0), (0, 0), (row_pad, row_pad), (col_pad, col_pad)))
    elif len(frames.shape) == 3:
        return np.pad(frames, ((0, 0), (row_pad, row_pad), (col_pad, col_pad)))
    else:
        return np.pad(frames, ((row_pad, row_pad), (col_pad, col_pad)))
