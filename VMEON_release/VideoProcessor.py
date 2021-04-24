import ffmpeg
import sys
import os
from .utilities import *


class VideoPreprocessor(object):
    """Video Preprocessor based on ffmpeg-python
        Parameters
        ----------
        video_file: string
            An absolute or relative path to the video file to be processed.
        width: int, optional
            Width of the input video. Only effective when video_file is a raw video, such as a YUV file.
            (Default: None)
        height: int, optional
            Height of the input video. Only effective when video_file is a raw video, such as a YUV file.
            (Default: None)
        fps: int, optional
            Frame rate of the input video. Only effective when video_file is a raw video, such as a YUV file.
            (Default: None)
        pix_fmt: int, optional
            Pixel format of the input video. Should be one of the FFmpeg-supported pixel formats.
            Check with "ffmpeg -pix_fmts" in terminal.
            Only effective when video_file is a raw video, such as a YUV file. (Default: None)
        """
    def __init__(self, video_file, width=None, height=None, fps=None, pix_fmt=None):
        if not os.path.exists(video_file):
            raise FileNotFoundError("{} not found!".format(video_file))
        self.input_dir, self.file_name = os.path.split(video_file)
        self.input_dir = os.path.abspath(self.input_dir)
        self.base_name, self.ext = os.path.splitext(self.file_name)
        self.file_name = os.path.join(self.input_dir, self.file_name)
        if self.ext in ['.yuv', '.raw']:
            if not all([width, height, fps, pix_fmt]):
                raise TypeError("Not enough information is provided for reading a raw video file!")
            else:
                self.width, self.height, self.fps, self.pix_fmt = width, height, fps, pix_fmt
                self.is_raw = True
            file_size = os.path.getsize(self.file_name)
            bpp = bpplut[self.pix_fmt][1]
            self.num_channels = bpplut[self.pix_fmt][0]
            self.num_frames = file_size / (self.width * self.height * (bpp / 8.0))
            if self.num_frames % 1 == 0:
                self.num_frames = int(self.num_frames)
            else:
                raise ValueError("Cannot get a valid total frame number with the given specs.")
            self.in_file = ffmpeg.input(self.file_name, s=str(self.width) + 'x' + str(self.height),
                                        r=self.fps, pix_fmt=self.pix_fmt)
        else:
            try:
                probe = ffmpeg.probe(self.file_name)
            except ffmpeg.Error as e:
                print(e.stderr, file=sys.stderr)
                sys.exit(1)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            self.is_raw = False
            self.width = int(video_stream['width'])
            self.height = int(video_stream['height'])
            self.fps = int(video_stream['r_frame_rate'].split('/')[0])
            self.pix_fmt = video_stream['pix_fmt']
            if 'nb_frames' in video_stream:
                self.num_frames = int(video_stream['nb_frames'])
            elif 'duration' in video_stream:
                self.num_frames = int(round(self.fps * float(video_stream['duration'])))
            else:
                self.num_frames = -1
                print('Cannot get total number of frames in the given video {}{}'.format(self.base_name, self.ext))
            self.num_channels = bpplut[self.pix_fmt][0]
            self.in_file = ffmpeg.input(self.file_name)

    def get_shape(self):
        return self.num_frames, self.height, self.width, self.num_channels
