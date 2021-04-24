import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from .VideoProcessor import VideoPreprocessor as Vp
# TODO: delete after debugging
# import matplotlib.pyplot as plt


class Video(Dataset):
    """Exploiting the parallel mechanism to accelerate video quality evaluation
    Parameters:
    ------------------------------------------
    video_path: string
        Path to the directory where frames of the test video are saved as png files,
        or the test video file.
    spatial_transform: function
        Transforms in spatial domain, such random cropping
    temporal_transform: function
        Transforms in temporal domain, such as random clipping. Not used for now
    segment_duration: int
        Indicating the number of frames in one clip.
        Should be compatible with the network input channel.
        Now set to 8.

    kwargs: only effective when the input video file is in raw video format, such as .yuv
    width: int
    height: int
    fps: int
    pix_fmt: string
        pix_fmt must be supported by ffmpeg
    """
    def __init__(self, video_path,
                 spatial_transform=None, temporal_transform=None,
                 segment_duration=None, normalize=255, **kwargs):
        if os.path.isdir(video_path):   # the video is saved as a folder of png files.
            self.loader = self._load_from_png
            self.num_frames = len([filename for filename in os.listdir(video_path)
                                   if filename.endswith('.png')])
        elif os.path.isfile(video_path):  # the video is saved as a video file.
            if video_path.endswith(('.yuv', '.raw')):  # raw video file
                try:
                    width = kwargs['width']
                    height = kwargs['height']
                    fps = kwargs['fps']
                    pix_fmt = kwargs['pix_fmt']
                except KeyError as key_err:
                    print('Raw video found. Lacking necessary video info.')
                    raise key_err
                self.vp = Vp(video_path, width=width, height=height, fps=fps, pix_fmt=pix_fmt)
            else:
                self.vp = Vp(video_path)   # other video files with meta info
            self.num_frames = self.vp.get_shape()[0]
            self.loader = self._load_from_video
        else:
            raise ValueError("{} is not a valid video path".format(video_path))

        self.clips = self._get_clips(segment_duration)
        self.video_path = video_path
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.normalize = normalize

    def _get_clips(self, sample_duration):
        clips = []
        step = sample_duration
        for i in range(0, self.num_frames - step + 1, step):
            clip_i = list(range(i, i + sample_duration))
            clips.append(clip_i)
        return clips

    def _load_from_png(self, frame_indices):
        """ load a clip from png image files
        Parameters:
        ------------------------------------------
        frame_indices: list of int
            A list of indexes indicating which frames to load

        Return:
        ------------------------------------------
        video: numpy ndarray
            A numpy array in the shape of (F, H, W, C) in the dtype of np.float32
        """

        video = []
        for idx in frame_indices:
            image_name = os.path.join(self.video_path, '{:d}.png'.format(idx + 1))
            pil_image = Image.open(image_name)
            np_image = np.array(pil_image, np.float32, copy=False)
            if pil_image.mode not in ('1', 'F'):
                np_image = np_image / self.normalize
            if len(np_image.shape) == 2:
                np_image = np_image[:, :, np.newaxis]
            video.append(np_image)
        video = np.stack(video, axis=0)
        return video

    def _load_from_video(self, frame_indices):
        """ load a clip from png image files
        Parameters:
        ------------------------------------------
        frame_indices: list of int
            A list of indexes indicating which frames to load

        Return:
        ------------------------------------------
        video: numpy ndarray
            A numpy array in the shape of (F, H, W, C) in the dtype of np.float32
        """
        start_frame = frame_indices[0]
        end_frame = start_frame + len(frame_indices)
        out, _ = (self.vp.in_file
                  .trim(start_frame=start_frame, end_frame=end_frame).filter_('extractplanes', 'y')
                  .output('pipe:', format='rawvideo', pix_fmt='gray')
                  .run(quiet=True)
                  )
        _, height, width, _ = self.vp.get_shape()
        video = (np.frombuffer(out, np.uint8).reshape((-1, height, width, 1)).astype(np.float32)) / self.normalize

        return video

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video clip.
        """
        frame_indices = self.clips[index]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(frame_indices)
        if self.spatial_transform is not None:
            clip = self.spatial_transform(clip)
        # video dimension: PCFHW
        clip = clip.contiguous()

        return clip

    def __len__(self):
        return len(self.clips)
