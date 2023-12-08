"""
Taken from by https://github.com/lisabug/guided-filter
File defining smoothing techniques for the guided filter: box filter and blurring
"""
import numpy as np
import cv2


def box_filter(I, r, normalize=True, border_type="reflect_101"):
    """

    Parameters
    ----------
    I: NDArray
        Input should be 3D with format of HWC
    r: int
        radius of filter. kernel size = 2 * r + 1
    normalize: bool
        Whether to normalize
    border_type: str
        Border type for padding, includes:
        edge        :   aaaaaa|abcdefg|gggggg
        zero        :   000000|abcdefg|000000
        reflect     :   fedcba|abcdefg|gfedcb
        reflect_101 :   gfedcb|abcdefg|fedcba

    Returns
    -------
    ret: NDArray
        Output has same shape with input
    """
    I = I.astype(np.float32)
    shape = I.shape
    assert len(shape) in [2, 3], \
        "I should be NDArray of 2D or 3D, not %dD" % len(shape)
    is_3D = True

    if len(shape) == 2:
        I = np.expand_dims(I, axis=2)
        shape = I.shape
        is_3D = False

    (rows, cols, channels) = shape

    tmp = np.zeros(shape=(rows, cols+2*r, channels), dtype=np.float32)
    ret = np.zeros(shape=shape, dtype=np.float32)

    # padding
    if border_type == 'reflect_101':
        res = I
        return cv2.boxFilter(src=I, dst=res, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), anchor=(-1,-1), normalize= normalize, borderType=cv2.BORDER_REFLECT_101)
    elif border_type == 'reflect':
        res = I
        return cv2.boxFilter(src=I, dst=res, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), anchor=(-1,-1), normalize= normalize, borderType=cv2.BORDER_REFLECT)
    elif border_type == 'edge':
        res = I
        return cv2.boxFilter(src=I, dst=res, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), anchor=(-1,-1), normalize= normalize, borderType=cv2.BORDER_REPLICATE)
    elif border_type == 'zero':
        res = I
        return cv2.boxFilter(src=I, dst=res, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), anchor=(-1,-1), normalize= normalize, borderType=cv2.BORDER_CONSTANT)
    else:
        raise NotImplementedError
