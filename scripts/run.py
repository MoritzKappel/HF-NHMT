#! /usr/bin/env python3

'''run.py: generates reenactment outputs for a set of pretrained networks from a source dataset.'''

import sys
import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np

import torch
import torch.optim as optim

import projectpath

with projectpath.context():
    import utils
    import config
    from Datasets import ActorDataset
    from Networks import SegmentationNetwork, StructureNetwork, RenderNetwork

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'


def main():

    # parse conmmand line arguments and load global config file
    parser = argparse.ArgumentParser(description='train.py: trains a new network instance.')
    parser.add_argument('-c', '--config', action='store', dest='config_path', default=None, metavar='PATH/TO/CONFIG_FILE/', required=True,
                        help='The config file to configure training.')

    args = parser.parse_args()
    config.loadConfig(args.config_path)

    # setup torch env
    tensor_type = utils.setupTorch(config.data.infer.use_cuda, config.data.infer.gpu_index)

    # create dataset
    requires_training_annotations = config.data.infer.use_gt_segmentation or config.data.infer.use_gt_structure
    source_dataset = ActorDataset(reenactment_mode=not requires_training_annotations)
    target_background = source_dataset.getBackground().type(tensor_type)

    # create networks
    utils.ColorLogger.print('Loading networks...', 'BOLD')
    segmentation_network = SegmentationNetwork().type(tensor_type)
    segmentation_optimizer = optim.Adam(segmentation_network.parameters(), 0.0)
    segmentation_network, segmentation_optimizer = utils.load_checkpoint(config.data.infer.segmentation_checkpoint, segmentation_network, segmentation_optimizer)
    structure_network = StructureNetwork().type(tensor_type)
    structure_optimizer = optim.Adam(structure_network.parameters(), 0.0)
    structure_network, structure_optimizer = utils.load_checkpoint(config.data.infer.structure_checkpoint, structure_network, structure_optimizer)
    render_network = RenderNetwork().type(tensor_type)
    render_optimizer = optim.Adam(render_network.parameters(), 0.0)
    render_network, render_optimizer = utils.load_checkpoint(config.data.infer.render_checkpoint, render_network, render_optimizer)

    # create output directories
    utils.ColorLogger.print('Creating output Directories...', 'BOLD')
    if os.path.exists(config.data.infer.output_dir):
        utils.ColorLogger.print("output directory already exists!", 'ERROR')
        sys.exit(0)
    os.makedirs(config.data.infer.output_dir, exist_ok=False)
    final_image_path = os.path.join(config.data.infer.output_dir, 'frames')
    os.makedirs(final_image_path, exist_ok=False)
    foreground_image_path = os.path.join(config.data.infer.output_dir, 'foreground')
    os.makedirs(foreground_image_path, exist_ok=False)
    if config.data.infer.generate_segmentations:
        segmentation_path = os.path.join(config.data.infer.output_dir, 'segmentation')
        os.makedirs(segmentation_path, exist_ok=False)
    if config.data.infer.generate_structure:
        structure_path = os.path.join(config.data.infer.output_dir, 'structure')
        os.makedirs(structure_path, exist_ok=False)
    if config.data.infer.create_videos:
        video_path = os.path.join(config.data.infer.output_dir, 'video')
        os.makedirs(video_path, exist_ok=False)

    # set modes
    segmentation_network.eval()
    structure_network.eval()
    render_network.eval()
    segmentation_network.clearState()
    structure_network.clearState()
    render_network.clearState()
    source_dataset.eval() if config.data.infer.validation_set else source_dataset.train()

    # create network outputs
    utils.ColorLogger.print('Generating reenactment outputs...', 'BOLD')
    with torch.no_grad():

        if config.data.infer.num_initial_iterations > 0:
            # refeed first frame to stabalize first zero-input
            images, poses, dynamics, segmentations, structure = utils.castDataTuple(source_dataset[0], tensor_type, unsqueeze=True)
            dynamics = torch.zeros_like(dynamics)  # set dynamics to zero (still frame)
            for _i in range(config.data.infer.num_initial_iterations):
                segmentation_output = torch.argmax(segmentation_network(poses, dynamics), dim=1, keepdim=True)
                input_segmentation = segmentations if config.data.infer.use_gt_segmentation else segmentation_output
                structure_output = structure_network(poses, dynamics, input_segmentation)
                input_structure = structure if config.data.infer.use_gt_structure else structure_output
                input_structure[:, 1] = torch.clamp(input_structure[:, 1] * config.data.infer.structure_magnification, min=0.0, max=1.0)
                foreground_output, final_output = render_network(poses, input_segmentation, input_structure, target_background)
        # loop over samples
        for sample_idx in tqdm(range(len(source_dataset)), desc="sample", leave=True):
            images, poses, dynamics, segmentations, structure = utils.castDataTuple(source_dataset[sample_idx], tensor_type, unsqueeze=True)
            # segmentation network
            segmentation_output = torch.argmax(segmentation_network(poses, dynamics), dim=1, keepdim=True)
            input_segmentation = segmentations if config.data.infer.use_gt_segmentation else segmentation_output
            # structure network
            structure_output = structure_network(poses, dynamics, input_segmentation)
            input_structure = structure if config.data.infer.use_gt_structure else structure_output
            input_structure[:, 1] = torch.clamp(input_structure[:, 1] * config.data.infer.structure_magnification, min=0.0, max=1.0)
            # render network
            foreground_output, final_output = render_network(poses, input_segmentation, input_structure, target_background)
            # save outputs
            final_output_numpy = (final_output.data[0].cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
            if config.data.infer.append_source_image:
                # resize and concatenate original frame
                images_numpy = (images.data[0].cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
                images_numpy = cv2.resize(images_numpy, (int(images_numpy.shape[1] * final_output_numpy.shape[0] / images_numpy.shape[0]), final_output_numpy.shape[0]))
                final_output_numpy = np.concatenate((images_numpy, final_output_numpy), axis=1)
            cv2.imwrite(os.path.join(final_image_path, '{0:05d}.png'.format(sample_idx)), final_output_numpy)
            cv2.imwrite(os.path.join(foreground_image_path, '{0:05d}.png'.format(sample_idx)), (foreground_output.data[0].cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0))
            if config.data.infer.generate_segmentations:
                cv2.imwrite(os.path.join(segmentation_path, '{0:05d}.png'.format(sample_idx)), (utils.visualizeSegmentationMap(segmentation_output[0], config.data.dataset.num_segmentation_labels).data.cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)[:, :, [2, 1, 0]])
            if config.data.infer.generate_structure:
                cv2.imwrite(os.path.join(structure_path, '{0:05d}.png'.format(sample_idx)), (utils.visualizeStructureMap(structure_output[0], tensor_type).data.cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)[:, :, [2, 1, 0]])
    utils.ColorLogger.print('Done!', 'BOLD')
    # create videos
    if config.data.infer.create_videos:
        utils.ColorLogger.print('Rendering videos using ffmpeg...', 'BOLD')
        video_data = [(final_image_path, 'final.mp4'), (foreground_image_path, 'foreground.mp4')]
        if config.data.infer.generate_segmentations:
            video_data.append((segmentation_path, 'segmentation.mp4'))
        if config.data.infer.generate_structure:
            video_data.append((structure_path, 'structure.mp4'))
        for i, j in video_data:
            command_string = "ffmpeg -framerate {1} -i {0} -c:v libx264 -pix_fmt yuv420p {2}".format(os.path.join(i, '%05d.png'), config.data.infer.video_framerate, os.path.join(video_path, j))
            if os.system(command_string) != 0:
                utils.ColorLogger.print('failed to create video using: "{0}"'.format(command_string), 'ERROR')
        utils.ColorLogger.print('Done!', 'BOLD')


if __name__ == '__main__':
    main()
