import math
import re
import os
import collections

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

from .VideoTransforms import ClipDenseSpatialCrop
from .Gdn import Gdn3D
from .Video import Video
from collections import Counter, OrderedDict

# TODO: delete after debugging
# import matplotlib.pyplot as plt

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class VmeonSlow(nn.Module):
    # Slow Fusion
    # input size: D = 8, H = 235, W = 235
    def __init__(self, output_channel):
        super(VmeonSlow, self).__init__()

        self.output_channel = output_channel
        # shared layer parameters

        self.conv1 = nn.Conv3d(1, 8, (2, 5, 5), stride=2, padding=(0, 2, 2))
        self.gdn1 = Gdn3D(8)
        self.conv2 = nn.Conv3d(8, 16, (2, 5, 5), stride=2, padding=(0, 2, 2))
        self.gdn2 = Gdn3D(16)
        self.conv3 = nn.Conv3d(16, 32, (2, 5, 5), stride=2, padding=(0, 2, 2))
        self.gdn3 = Gdn3D(32)
        self.conv4 = nn.Conv3d(32, 64, (1, 3, 3), stride=1, padding=0)
        self.gdn4 = Gdn3D(64)

        # subtask 1 parameters
        self.st1_fc1 = nn.Conv3d(64, 128, 1, stride=1, padding=0)
        self.st1_gdn1 = Gdn3D(128)
        self.st1_fc2 = nn.Conv3d(128, self.output_channel, 1, stride=1, padding=0)

        # subtask 2 parameters
        self.st2_fc1 = nn.Conv3d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn1 = Gdn3D(256)
        self.st2_fc2 = nn.Conv3d(256, self.output_channel, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn3D):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        # shared layer
        x = F.max_pool3d(self.gdn1(self.conv1(x)), (1, 2, 2))
        x = F.max_pool3d(self.gdn2(self.conv2(x)), (1, 2, 2))
        x = F.max_pool3d(self.gdn3(self.conv3(x)), (1, 2, 2))
        x = F.max_pool3d(self.gdn4(self.conv4(x)), (1, 2, 2))

        # subtask 1
        y1 = self.st1_fc2(self.st1_gdn1(self.st1_fc1(x)))
        y = y1.view(batch_size, -1)

        # subtask 2
        p = F.softmax(y1, dim=1)
        y2 = self.st2_gdn1(self.st2_fc1(x))
        s = self.st2_fc2(y2)
        s = s.view(batch_size, -1)
        p = p.view(batch_size, -1)
        g = torch.sum(p * s, dim=1)
        return y, g


class Predictor:
    def __init__(self, config):
        # Transforms and training database
        self.num_workers = config.num_workers
        self.frame_per_clip = 8
        self.use_cuda = config.use_cuda

        self.test_spatial_transform = transforms.Compose([
            ClipDenseSpatialCrop(output_size=235, stride=config.stride)])

        self.test_temporal_transform = None

        # initialize the model
        self.model = VmeonSlow(4)
        self.model_name = type(self.model).__name__
        print(self.model)

        if torch.cuda.device_count() > 1 and config.use_cuda:
            print("[*] GPU #", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=[0])

        if torch.cuda.is_available() and config.use_cuda:
            self.model.cuda()

        if os.path.isfile(config.ckpt):
            self.ckpt = config.ckpt
        else:
            vmeon_path = os.path.dirname(__file__)
            self.ckpt = os.path.join(vmeon_path, 'VMEON_release.pt')

        # try load the model
        self._load_checkpoint(ckpt=self.ckpt)

    def predict_quality(self, video_path, **kwargs):
        data = Video(video_path, spatial_transform=self.test_spatial_transform,
                     temporal_transform=self.test_temporal_transform,
                     segment_duration=self.frame_per_clip, **kwargs)

        data_loader = DataLoader(data, batch_size=1,
                                 shuffle=False,
                                 collate_fn=self._collate_f2b,
                                 num_workers=self.num_workers)

        pred_ds = []
        pred_qs = []
        for i, clip in enumerate(data_loader):
            # print(i)
            inputs = Variable(clip)
            if self.use_cuda:
                inputs = inputs.cuda()

            y, q = self.model(inputs)

            y = y.cpu().data
            _, max_indices = y.max(dim=1)
            max_idx_c = self._robust_mode(max_indices)
            pred_ds.append(max_idx_c)

            q = q.cpu().data
            pred_qs.append(torch.mean(q))

        pred_disttype = self._robust_mode(pred_ds).numpy()
        pred_quality = torch.mean(torch.stack(pred_qs)).numpy()

        return pred_disttype, pred_quality

    @staticmethod
    def _robust_mode(L):
        """
        :param L: Iterable
        :return: mode of L. If multiple modes, return one of them randomly
        """
        c = Counter(L)
        return c.most_common()[0][0]

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            if not isinstance(self.model, nn.DataParallel):
                state_dict = self._remove_prefix(state_dict)
            self.model.load_state_dict(state_dict)
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _remove_prefix(state_dict):
        tmp_list = []
        for k, v in state_dict.items():
            sub_strings = k.split('.')
            new_key = '.'.join(sub_strings[1:])
            tmp_list.append((new_key, v))

        new_state_dict = OrderedDict(tmp_list)
        return new_state_dict

    # collate frames to batch
    def _collate_f2b(self, batch):
        "collate num dim (and crop dim) into batch dim."
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if torch.is_tensor(batch[0]):
            out = torch.stack(batch, 0)
            # BNCHW
            if out.dim() == 5:
                B, N, C, H, W = list(out.size())

                # (BN)CHW
                out = out.view(-1, C, H, W)
            # BNCFHW
            elif out.dim() == 6:
                B, N, C, F, H, W = list(out.size())

                # (BN)CFHW
                out = out.view(-1, C, F, H, W)
            return out
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], int):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], (str, bytes)):
            return batch
        elif isinstance(batch[0], collections.Mapping):
            return {key: self._collate_f2b([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [self._collate_f2b(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))
