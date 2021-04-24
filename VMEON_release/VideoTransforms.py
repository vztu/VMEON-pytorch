import numpy as np
import torch


class ClipDenseSpatialCrop(object):
    """Dense spatial cropper for a clip consisting of several consecutive frames.
    Parameters:
    ------------------
    output_size: tuple or int
        Desired output size. If int, square crop is made.

    stride: tuple or int, optional
        Stride of cropped patches.
        If not specified, cropped patches will be non-overlapping, i.e. stride = output_size.
    """

    def __init__(self, output_size, stride=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        if not stride:
            self.stride = self.output_size
        else:
            assert isinstance(stride, (int, tuple))
            if isinstance(stride, int):
                self.stride = (stride, stride)
            else:
                assert len(stride) == 2
                self.stride = stride

    def __call__(self, clip):
        """Crop a clip spatially into several smaller clips, whose spatial size is specified by output_size
        Parameters
        -------------------------------
        clip: np.ndarray
            a numpy array with a shape of (F, H, W, C)

        Return
        -------------------------------
        patches: torch.tensor
            a Torch tensor in the shape of (P, C, F, *self.output_size)
        """
        f, h, w, c = clip.shape
        new_h, new_w = self.output_size
        stride_h, stride_w = self.stride

        h_start = np.arange(0, h - new_h, stride_h)
        w_start = np.arange(0, w - new_w, stride_w)

        patches = [clip[:, hv_s:hv_s + new_h, wv_s:wv_s + new_w, :] for hv_s in h_start for wv_s in w_start]
        patches = np.stack(patches, axis=0)
        # From PFHWC to PCFHW
        patches = patches.transpose((0, 4, 1, 2, 3))

        patches = torch.from_numpy(patches)
        return patches