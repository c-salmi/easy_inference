import numpy as np

def pad_frames(frames, width, height):
    row_pad, col_pad = ([width, height] - np.array(frames.shape)[-2:])//2

    if len(frames.shape) == 4:
        assert row_pad >= 0 and col_pad >= 0
        rgb_frames = np.pad(frames, (
            (0, 0),
            (0, 0),
            (row_pad, row_pad),
            (col_pad, col_pad) 
        ))
        return rgb_frames
    else:
        depth_frames = np.pad(frames, (
            (0, 0),
            (row_pad, row_pad),
            (col_pad, col_pad) 
        ))
        return depth_frames
