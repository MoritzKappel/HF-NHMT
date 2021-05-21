'''Networks.py: pytorch ANN models.'''

import numpy as np
import scipy.ndimage

import torch
import torch.nn as nn

import config
import utils

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'


class ConvBlock(nn.Module):
    def __init__(self, skip_connection, num_input, num_output, kernel_size, kernel_stride):
        super(ConvBlock, self).__init__()
        self.skip_connection = skip_connection
        # create block
        self.block = nn.Sequential(nn.Conv2d(num_input, num_output, kernel_size=kernel_size, stride=kernel_stride, padding=kernel_size//2, bias=True, padding_mode='reflect'),
                                   nn.InstanceNorm2d(num_output),
                                   nn.ReLU(inplace=True))

    def forward(self, inputs):
        x, skip_list, index = inputs
        y = self.block(x)
        if self.skip_connection:
            skip_list.append(y)
            index += 1
        return (y, skip_list, index)


class TranspConvBlock(nn.Module):
    def __init__(self, skip_connection, num_input, num_output, kernel_size, kernel_stride=2):
        super(TranspConvBlock, self).__init__()
        self.skip_connection = skip_connection
        self.block = nn.Sequential(nn.ConvTranspose2d(num_input, num_output, kernel_size=kernel_size, stride=kernel_stride, padding=kernel_size//2, output_padding=1, bias=True),
                                   nn.InstanceNorm2d(num_output),
                                   nn.ReLU(inplace=True))

    # helper function to fit dimensions if input is not power-of-2
    def fit_dimensions(self, source, target):
        return source if source.size()[2:] == target.size()[2:] else source[:, :, :target.size(2), :target.size(3)]

    def forward(self, inputs):
        x, skip_list, index = inputs
        if self.skip_connection:
            index -= 1
            x = torch.cat((self.fit_dimensions(x, skip_list[index]), skip_list[index]), dim=1)
        y = self.block(x)
        return (y, skip_list, index)


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        # create block
        self.block = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True, padding_mode='reflect'),
                                   nn.InstanceNorm2d(num_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True, padding_mode='reflect'),
                                   nn.InstanceNorm2d(num_channels))

    def forward(self, inputs):
        x, skip_list, index = inputs
        return (x + self.block(x), skip_list, index)


# transforms input modalities from pytorch modules to custom blocks
class BeginCustomBlocks(nn.Module):
    def __init__(self):
        super(BeginCustomBlocks, self).__init__()

    def forward(self, inputs):
        return (inputs, [], 0)


# transforms input modalities from custom blocks back to pytorch modules
class EndCustomBlocks(nn.Module):
    def __init__(self):
        super(EndCustomBlocks, self).__init__()

    def forward(self, inputs):
        x, skip_list, index = inputs
        if index != 0:
            utils.ColorLogger.print('Unused skip-connections detected after end of custom block!', 'WARNING')
        return x


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        # create layers
        num_input_cannels = (5 * config.data.dataset.num_bone_channels) + config.data.dataset.num_segmentation_labels  # pose, dynamics, last seg binarized
        num_output_channels = config.data.dataset.num_segmentation_labels
        self.layers = nn.Sequential(
            BeginCustomBlocks(),
            # initial encoder
            ConvBlock(False, num_input_cannels, 64, 7, 1),
            # downsampling blocks
            ConvBlock(True, 64, 128, 3, 2),
            ConvBlock(True, 128, 256, 3, 2),
            ConvBlock(True, 256, 512, 3, 2),
            ConvBlock(False, 512, 1024, 3, 2),
            # residual blocks
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            # upsampling blocks
            TranspConvBlock(False, 1024, 512, 3, 2),
            TranspConvBlock(True, 1024, 256, 3, 2),
            TranspConvBlock(True, 512, 128, 3, 2),
            TranspConvBlock(True, 256, 64, 3, 2),
            EndCustomBlocks(),
            # final conv & activation
            nn.Conv2d(64, num_output_channels, kernel_size=7, stride=1, padding=3, bias=True, padding_mode='reflect')
        )
        self.clearState()

    # reset last output
    def clearState(self):
        self.last_state = torch.zeros((1, config.data.dataset.num_segmentation_labels, config.data.dataset.target_actor_height, config.data.dataset.target_actor_width), requires_grad=False)

    # return number of trainable parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pose, dynamics, last_state=None):
        # generate outputs
        output = self.layers(torch.cat((pose, dynamics, self.softmax(self.last_state)), dim=1))
        # store last state
        self.last_state = utils.segmentationLabelsToChannels(torch.argmax(output, dim=1, keepdim=True).detach() if last_state is None else last_state, config.data.dataset.num_segmentation_labels)
        # return
        return output


class StructureNetwork(nn.Module):
    def __init__(self):
        super(StructureNetwork, self).__init__()
        # create layers
        num_input_cannels = (5 * config.data.dataset.num_bone_channels) + config.data.dataset.num_segmentation_labels + 2  # pose, dynamics, seg channels, last structure
        num_output_channels = 2
        self.layers = nn.Sequential(
            BeginCustomBlocks(),
            # initial encoder
            ConvBlock(False, num_input_cannels, 64, 7, 1),
            # downsampling blocks
            ConvBlock(True, 64, 128, 3, 2),
            ConvBlock(True, 128, 256, 3, 2),
            ConvBlock(True, 256, 512, 3, 2),
            ConvBlock(False, 512, 1024, 3, 2),
            # residual blocks
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            # upsampling blocks
            TranspConvBlock(False, 1024, 512, 3, 2),
            TranspConvBlock(True, 1024, 256, 3, 2),
            TranspConvBlock(True, 512, 128, 3, 2),
            TranspConvBlock(True, 256, 64, 3, 2),
            EndCustomBlocks(),
            # final conv & activation
            nn.Conv2d(64, num_output_channels, kernel_size=7, stride=1, padding=3, bias=True, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.clearState()

    # reset last output
    def clearState(self):
        self.last_state = torch.zeros((1, 2, config.data.dataset.target_actor_height, config.data.dataset.target_actor_width), requires_grad=False)

    # return number of trainable parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pose, dynamics, segmentation_labels, last_state=None):
        # generate outputs
        clothes_mask = segmentation_labels < 0
        for label in config.data.dataset.segmentation_clothes_labels:
            clothes_mask = torch.logical_or(clothes_mask, segmentation_labels == label)
        output = clothes_mask * self.layers(torch.cat((pose, dynamics, utils.segmentationLabelsToChannels(segmentation_labels, config.data.dataset.num_segmentation_labels), self.last_state), dim=1))
        # store last state
        self.last_state = output.detach() if last_state is None else last_state
        # return
        return output


class RenderNetwork(nn.Module):
    def __init__(self):
        super(RenderNetwork, self).__init__()
        # create layers
        num_foreground_input_cannels = config.data.dataset.num_bone_channels + config.data.dataset.num_segmentation_labels + 2 + 3  # pose, segmentation channels, structure, last image
        num_foreground_output_channels = 3
        self.foreground_layers = nn.Sequential(
            BeginCustomBlocks(),
            # initial encoder
            ConvBlock(False, num_foreground_input_cannels, 64, 7, 1),
            # downsampling blocks
            ConvBlock(True, 64, 128, 3, 2),
            ConvBlock(True, 128, 256, 3, 2),
            ConvBlock(True, 256, 512, 3, 2),
            ConvBlock(False, 512, 1024, 3, 2),
            # residual blocks
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            # upsampling blocks
            TranspConvBlock(False, 1024, 512, 3, 2),
            TranspConvBlock(True, 1024, 256, 3, 2),
            TranspConvBlock(True, 512, 128, 3, 2),
            TranspConvBlock(True, 256, 64, 3, 2),
            EndCustomBlocks(),
            # final conv & activation
            nn.Conv2d(64, num_foreground_output_channels, kernel_size=7, stride=1, padding=3, bias=True, padding_mode='reflect'),
            nn.Sigmoid()
        )
        num_combined_input_cannels = 3 + config.data.dataset.num_segmentation_labels  # fg_bg_fusion + segmentation channels
        num_combined_output_channels = 3  # final image
        self.combined_layers = nn.Sequential(
            # initial encoder
            BeginCustomBlocks(),
            ConvBlock(False, num_combined_input_cannels, 64, 7, 1),
            # downsampling blocks
            ConvBlock(True, 64, 128, 3, 2),
            ConvBlock(True, 128, 256, 3, 2),
            ConvBlock(True, 256, 512, 3, 2),
            ConvBlock(False, 512, 1024, 3, 2),
            # residual blocks
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            ResidualBlock(1024, 3),
            # upsampling blocks
            TranspConvBlock(False, 1024, 512, 3, 2),
            TranspConvBlock(True, 1024, 256, 3, 2),
            TranspConvBlock(True, 512, 128, 3, 2),
            TranspConvBlock(True, 256, 64, 3, 2),
            EndCustomBlocks(),
            # final conv & activation
            nn.Conv2d(64, num_combined_output_channels, kernel_size=7, stride=1, padding=3, bias=True, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.clearState()

    # reset last output
    def clearState(self):
        self.last_state = torch.zeros((1, 3, config.data.dataset.target_actor_height, config.data.dataset.target_actor_width), requires_grad=False)

    # for wgan
    def clampWeights(self, clamp_val):
        for p in self.parameters():
            p.data.clamp_(-clamp_val, clamp_val)

    # return number of trainable parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, pose, segmentation_labels, structure, background, last_state=None):
        # segmentation labels to channels
        segmentation_channels = utils.segmentationLabelsToChannels(segmentation_labels, config.data.dataset.num_segmentation_labels)
        binary_fg_mask = 1.0 - segmentation_channels[:, config.data.dataset.segmentation_background_label].unsqueeze(1)
        # generate foreground outputs
        output_foreground = binary_fg_mask * self.foreground_layers(torch.cat((pose, segmentation_channels, structure, self.last_state), dim=1))
        # store last state
        self.last_state = output_foreground.detach() if last_state is None else last_state * binary_fg_mask
        # generate final image by foreground-background blending
        combined_image = output_foreground.detach() + ((1.0 - binary_fg_mask) * background)
        output_final = self.combined_layers(torch.cat((combined_image, segmentation_channels), dim=1))
        # return
        return output_foreground + ((1.0 - binary_fg_mask) * torch.ones_like(output_foreground)), output_final


# apply Gaussian blur to torch tensor (adapted from https://discuss.pytorch.org/t/gaussian-kernel-layer/37619)
class GaussianBlur(nn.Module):
    def __init__(self, num_channels, kernel_size, std):
        super(GaussianBlur, self).__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=kernel_size//2, bias=None, groups=num_channels)
        self.weights_init(num_channels, kernel_size, std)

    def forward(self, x, unsqueeze=True):
        return self.conv(x.unsqueeze(0)).squeeze() if unsqueeze else self.conv(x)

    def weights_init(self, num_channels, kernel_size, std):
        n = np.zeros((kernel_size, kernel_size), np.float32)
        n[kernel_size // 2, kernel_size // 2] = 1.0
        k = scipy.ndimage.gaussian_filter(n, sigma=std)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
