'''Utils.py: miscellaneous utility functions and classes.'''

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import natsort

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'

SUPPORTED_TORCH_VERSION = '1.5.1'
SUPPORTED_CV_VERSION = '4.5.2'
SUPPORTED_CUDA_VERSION = '10.2'


class ColorLogger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print(msg, msg_type):
        print('{0}{1}{2}'.format(ColorLogger.__dict__[msg_type], msg, ColorLogger.ENDC) if msg_type not in ('ERROR', 'WARNING') else '{0}{1}:{2} {3}'.format(ColorLogger.__dict__[msg_type], msg_type, ColorLogger.ENDC, msg))


class AverageMeter(object):
    # AverageMeter.py: AverageMeter class extracted from the Pytoch Imagenet examples (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def checkLibraryVersions():
    if torch.__version__ != SUPPORTED_TORCH_VERSION:
        ColorLogger.print('current Pytorch version: {0} (tested with {1})'.format(torch.__version__, SUPPORTED_TORCH_VERSION), 'WARNING')
    if torch.version.cuda != SUPPORTED_CUDA_VERSION:
        ColorLogger.print('current (Pytorch) CUDA version: {0} (tested with {1})'.format(torch.version.cuda, SUPPORTED_CUDA_VERSION), 'WARNING')
    if cv2.__version__ != SUPPORTED_CV_VERSION:
        ColorLogger.print('current OpenCV version: {0} (tested with {1})'.format(cv2.__version__, SUPPORTED_CV_VERSION), 'WARNING')


def setupTorch(USE_CUDA, GPU_INDEX):
    # print library version mismatches
    checkLibraryVersions()
    # set default params + enable gpu
    tensor_type = 'torch.FloatTensor'
    if USE_CUDA and torch.cuda.is_available():
        torch.cuda.set_device(GPU_INDEX)
        tensor_type = 'torch.cuda.FloatTensor'
        torch.set_default_tensor_type(tensor_type)
        cudnn.benchmark = True
        cudnn.fastest = True
    # additional debug stuff
    torch.utils.backcompat.broadcast_warning.enabled = True
    # uncomment to fix potential opencv-pytorch multiprocessing issues
    # multiprocessing.set_start_method('spawn', force=True)
    return tensor_type


def castDataTuple(t, tensor_type, unsqueeze=False):
    func = lambda x: (x.type(tensor_type) if not unsqueeze else x.type(tensor_type).unsqueeze(0)) if x is not None else x
    return tuple(func(item) for item in t)


def list_sorted_files(path):
    return natsort.natsorted([file_name for file_name in os.listdir(path) if os.path.isfile(os.path.join(path, file_name))])


def list_sorted_directories(path):
    return natsort.natsorted([file_name for file_name in os.listdir(path) if not os.path.isfile(os.path.join(path, file_name))])


def save_checkpoint(net_g, optimizer_g, checkpoint_path):
    torch.save({'net_g_state_dict': net_g.state_dict() if net_g is not None else None,
                'optimizer_g_state_dict': optimizer_g.state_dict() if optimizer_g is not None else None},
               checkpoint_path)


def load_checkpoint(checkpoint_path, net_g, optimizer_g, map_location=lambda storage, location: storage):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if checkpoint['net_g_state_dict'] is not None and net_g is not None:
        net_g.load_state_dict(checkpoint['net_g_state_dict'])
    if checkpoint['optimizer_g_state_dict'] is not None and optimizer_g is not None:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    del checkpoint
    torch.cuda.empty_cache()
    return net_g, optimizer_g


def visualizeSegmentationMap(tensor, num_labels):
    # calculate color palette, adopted from "https://github.com/PeikeLi/Self-Correction-Human-Parsing/blob/master/evaluate.py" for better comparison
    palette = torch.zeros((num_labels, 3)).byte()
    for j in range(num_labels):
        lab = j
        i = 0
        while lab:
            palette[j, 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j, 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j, 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    palette = palette.type_as(tensor) / 255.0
    palette[0, 0] = 1.0
    palette[0, 1] = 1.0
    palette[0, 2] = 1.0
    # apply palette
    return palette[tensor[0].long()].permute(2, 0, 1)


def segmentationLabelsToChannels(tensor, num_labels):
    out = torch.zeros((1, num_labels, tensor.shape[2], tensor.shape[3]), dtype=torch.float32, layout=tensor.layout, device=tensor.device)
    out.scatter_(1, tensor.long(), value=1.0)
    return out.float()


# visualize structure conditioning similar to optical flow
def visualizeStructureMap(tensor, tensor_type):
    tensor_numpy = tensor.data.cpu().numpy().astype(np.float32)
    tensor_numpy[0] *= 360.0
    tensor_rbg = cv2.cvtColor(np.concatenate((tensor_numpy, np.ones((1, tensor_numpy.shape[1], tensor_numpy.shape[2]), dtype=tensor_numpy.dtype)), axis=0).transpose((1, 2, 0)), cv2.COLOR_HSV2RGB)
    return torch.from_numpy(tensor_rbg.transpose((2, 0, 1))).type(tensor_type)


# generates rbg image from 2xHxW float32 tensors
def pseudocolorOpticalFlowNumpy(flow, norm_factor=None):
    (x, y) = (np.array(flow[0], copy=True), np.array(flow[1], copy=True))
    (magnitude, angle) = cv2.cartToPolar(x, y, angleInDegrees=True)
    if norm_factor is None:
        cv2.normalize(magnitude, magnitude, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    else:
        magnitude = np.divide(magnitude, norm_factor)
        ret, magnitude = cv2.threshold(magnitude, 1.0, 0, cv2.THRESH_TRUNC)
    hsv = cv2.merge((angle, magnitude, np.ones(angle.shape, dtype=angle.dtype)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# sets the given optimizers default learning rate (for all parameters)
def setOptimizerLR(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
