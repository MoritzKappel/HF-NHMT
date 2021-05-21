'''Losses.py: Loss functions.'''

import sys

import torch
import torch.nn as nn
import torchvision.models as models

import config
import utils

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'


class PerceptiveLoss(nn.Module):

    def __init__(self):
        super(PerceptiveLoss, self).__init__()
        self.layers = config.data.train.vgg_layers
        self.vgg = models.vgg19(pretrained=True)
        self.l1 = nn.L1Loss()
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        self.std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def exec_vgg(self, x):
        outputs = []
        for i in range(len(self.layers)):
            outputs.append(self.vgg.features[0 if i == 0 else self.layers[i-1]+1:self.layers[i]+1](x if not outputs else outputs[-1]))
        return torch.cat(list(t.flatten(start_dim=1) for t in outputs), 1)

    def forward(self, img_out, img_target):
        self.vgg.eval()
        return self.l1(self.exec_vgg((img_out-self.mean)/self.std), self.exec_vgg((img_target-self.mean)/self.std))


class PrintableLoss(nn.Module):
    def __init__(self, name):
        super(PrintableLoss, self).__init__()
        self.name = name
        # loss visualization objects
        self.accumulators = (utils.AverageMeter(), utils.AverageMeter())

    def accumulateLoss(self, loss, num_instances):
        self.accumulators[1 if not self.training else 0].update(loss.item(), num_instances)

    def printLoss(self, writer, epoch, clear=True):
        writer.add_scalars(self.name, {'train': self.accumulators[0].avg, 'eval': self.accumulators[1].avg}, epoch)
        if clear:
            self.clearLoss()

    def clearLoss(self):
        for x in self.accumulators:
            x.reset()


class SegmentationLoss(PrintableLoss):
    def __init__(self):
        super(SegmentationLoss, self).__init__('segmentation_loss')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, output_segmentation, gt_segmentation):
        # calculate & return losses
        l_segmentation = self.ce(output_segmentation, gt_segmentation[:, 0].long()) * config.data.train.segmentation_net.loss_lambda
        self.accumulateLoss(l_segmentation, gt_segmentation.size(0))
        return l_segmentation


class StructureLoss(PrintableLoss):
    def __init__(self):
        super(StructureLoss, self).__init__('structure_loss')
        self.reconstruction = nn.L1Loss(reduction='none')

    def forward(self, output_structure, gt_structure, gt_segmentation):
        # create cothes maskth
        clothes_mask = gt_segmentation < 0
        for label in config.data.dataset.segmentation_clothes_labels:
            clothes_mask = torch.logical_or(clothes_mask, gt_segmentation == label)

        # calculate & return losses
        l_structure = (self.reconstruction(output_structure, gt_structure) * clothes_mask).mean() * config.data.train.structure_net.loss_lambda
        self.accumulateLoss(l_structure, gt_structure.size(0))
        return l_structure


class RenderLoss(PrintableLoss):
    def __init__(self):
        super(RenderLoss, self).__init__('render_loss')
        self.vgg = PerceptiveLoss()
        self.reconstruction = nn.L1Loss()

    def forward(self, output_foreground, output_final, gt_image, gt_segmentation):
        # calculate & return losses
        masked_foreground = gt_image * torch.where(gt_segmentation == config.data.dataset.segmentation_background_label, torch.tensor([0.0]), torch.tensor([1.0]))
        l_reconstruction_final = (self.vgg(output_final, gt_image) * config.data.train.render_net.loss_lambda_final_perceptive) + (self.reconstruction(output_final, gt_image) * config.data.train.render_net.loss_lambda_final_reconstruction)
        l_reconstruction_foreground = ((self.vgg(output_foreground, masked_foreground) * config.data.train.render_net.loss_lambda_foreground_perceptive) + (self.reconstruction(output_foreground, masked_foreground) * config.data.train.render_net.loss_lambda_foreground_reconstruction))
        l_total = l_reconstruction_final + l_reconstruction_foreground
        self.accumulateLoss(l_total, gt_image.size(0))
        return l_total


# calculate spatial gradient of input Tensor
class SpatialGradient(nn.Module):
    def __init__(self, num_channels, order=1):
        super(SpatialGradient, self).__init__()
        if order not in [1, 2]:
            utils.ColorLogger.print('SpatialGradientLoss currently only supports 1st and 2nd order smoothness!', 'ERROR')
            sys.exit(0)
        # create convs with fixed custom kernel
        kernel_data_x = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]] if order == 2 else [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        kernel_data_y = torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]] if order == 2 else [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        conv_kernel_x = torch.zeros((num_channels, num_channels, 3, 3))
        conv_kernel_y = torch.zeros((num_channels, num_channels, 3, 3))
        for c in range(num_channels):
            conv_kernel_x[c, c] = kernel_data_x
            conv_kernel_y[c, c] = kernel_data_y
        self.conv_x = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(conv_kernel_x)
        self.conv_x.weight.requires_grad = False
        self.conv_y = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = nn.Parameter(conv_kernel_y)
        self.conv_y.weight.requires_grad = False

    def forward(self, input_tensor, direction=0):
        # direction: <0 = x, >0 = y, 0 = norm_2(x,y)
        if direction < 0:
            out_grad = self.conv_x(input_tensor)
        elif direction > 0:
            out_grad = self.conv_y(input_tensor)
        else:
            grad_x = self.conv_x(input_tensor)
            grad_y = self.conv_y(input_tensor)
            out_grad = torch.norm(torch.stack([grad_x, grad_y], 0), 2, 0)

        return torch.abs(out_grad)
