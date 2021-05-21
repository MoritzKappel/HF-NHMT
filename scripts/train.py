#! /usr/bin/env python3

'''train.py: train a new network.'''

import os
import argparse
import datetime
from tqdm import tqdm
from collections import namedtuple

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import projectpath

with projectpath.context():
    import utils
    import config
    from Datasets import ActorDataset
    from Networks import SegmentationNetwork, StructureNetwork, RenderNetwork
    from Losses import SegmentationLoss, StructureLoss, RenderLoss

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'


def main():
    # parse conmmand line arguments and load global config file
    parser = argparse.ArgumentParser(description='train.py: trains a new network instance.')
    parser.add_argument('-c', '--config', action='store', dest='config_path', default=None, metavar='PATH/TO/config_FILE/', required=True, help='The config file to configure training.')
    args = parser.parse_args()
    config.loadConfig(args.config_path)

    # setup torch env
    tensor_type = utils.setupTorch(config.data.train.use_cuda, config.data.train.gpu_index)

    # create dataset
    dataset = ActorDataset()
    background = dataset.getBackground().type(tensor_type)

    # create networks, optimizers, schedulers and criterions for every requested
    training_instance = namedtuple('training_instance', ['network', 'optimizer', 'scheduler', 'criterion'])
    training_instances = {}

    if config.data.train.segmentation_net.enable:
        segmentation_network = SegmentationNetwork().type(tensor_type)
        segmentation_optimizer = optim.Adam(segmentation_network.parameters(), config.data.train.segmentation_net.learningrate, betas=(config.data.train.segmentation_net.adam_beta1, config.data.train.segmentation_net.adam_beta2))
        segmentation_scheduler = optim.lr_scheduler.StepLR(segmentation_optimizer, step_size=config.data.train.segmentation_net.learningrate_decay_step, gamma=config.data.train.segmentation_net.learningrate_decay_factor)
        segmentation_criterion = SegmentationLoss().type(tensor_type)
        # load checkpoint (if available)
        if config.data.train.segmentation_net.initial_checkpoint is not None:
            segmentation_network, segmentation_optimizer = utils.load_checkpoint(config.data.train.segmentation_net.initial_checkpoint, segmentation_network, segmentation_optimizer)
        training_instances['segmentation'] = training_instance(network=segmentation_network, optimizer=segmentation_optimizer, scheduler=segmentation_scheduler, criterion=segmentation_criterion)

    if config.data.train.structure_net.enable:
        structure_network = StructureNetwork().type(tensor_type)
        structure_optimizer = optim.Adam(structure_network.parameters(), config.data.train.structure_net.learningrate, betas=(config.data.train.structure_net.adam_beta1, config.data.train.structure_net.adam_beta2))
        structure_scheduler = optim.lr_scheduler.StepLR(structure_optimizer, step_size=config.data.train.structure_net.learningrate_decay_step, gamma=config.data.train.structure_net.learningrate_decay_factor)
        structure_criterion = StructureLoss().type(tensor_type)
        # load checkpoint (if available)
        if config.data.train.structure_net.initial_checkpoint is not None:
            structure_network, structure_optimizer = utils.load_checkpoint(config.data.train.structure_net.initial_checkpoint, structure_network, structure_optimizer)
        training_instances['structure'] = training_instance(network=structure_network, optimizer=structure_optimizer, scheduler=structure_scheduler, criterion=structure_criterion)

    if config.data.train.render_net.enable:
        render_network = RenderNetwork().type(tensor_type)
        render_optimizer = optim.Adam(render_network.parameters(), config.data.train.render_net.learningrate, betas=(config.data.train.render_net.adam_beta1, config.data.train.render_net.adam_beta2))
        render_scheduler = optim.lr_scheduler.StepLR(render_optimizer, step_size=config.data.train.render_net.learningrate_decay_step, gamma=config.data.train.render_net.learningrate_decay_factor)
        render_criterion = RenderLoss().type(tensor_type)
        # load checkpoint (if available)
        if config.data.train.render_net.initial_checkpoint is not None:
            render_network, render_optimizer = utils.load_checkpoint(config.data.train.render_net.initial_checkpoint, render_network, render_optimizer)
        training_instances['render'] = training_instance(network=render_network, optimizer=render_optimizer, scheduler=render_scheduler, criterion=render_criterion)

    # IO: create visualization objects and checkpoint output directory
    training_name_prefix = '{0}{date:%Y-%m-%d-%H:%M:%S}'.format(config.data.train.name_prefix + '_' if config.data.train.name_prefix is not None else '', date=datetime.datetime.now())
    checkpoint_output_path = os.path.join(config.data.train.output_checkpoints_dir, training_name_prefix)
    os.makedirs(checkpoint_output_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.data.train.tensorboard_logdir, training_name_prefix))
    utils.ColorLogger.print('training name prefix: ' + training_name_prefix, 'OKBLUE')

    # visualize config and samples in tensorboard
    config.printConfigToTensorboard(args.config_path, writer)
    dataset.train()
    training_visualization_indices = [int((i * len(dataset)) + 0.5) for i in config.data.train.training_visualization_indices]
    for i in training_visualization_indices:
        images, poses, dynamics, segmentations, structure = utils.castDataTuple(dataset[i], tensor_type, unsqueeze=False)
        dataset.visualizeSample(writer, 'training_sample', 0, [images[[2, 1, 0]], background[0, [2, 1, 0]], poses.sum(dim=0, keepdim=True), utils.visualizeSegmentationMap(segmentations, config.data.dataset.num_segmentation_labels), utils.visualizeStructureMap(structure, tensor_type)])
    dataset.eval()
    validation_visualization_indices = [int((i * len(dataset)) + 0.5) for i in config.data.train.validation_visualization_indices]
    for i in validation_visualization_indices:
        images, poses, dynamics, segmentations, structure = utils.castDataTuple(dataset[i], tensor_type, unsqueeze=False)
        dataset.visualizeSample(writer, 'validation_sample', 0, [images[[2, 1, 0]], background[0, [2, 1, 0]], poses.sum(dim=0, keepdim=True), utils.visualizeSegmentationMap(segmentations, config.data.dataset.num_segmentation_labels), utils.visualizeStructureMap(structure, tensor_type)])

    # mainloop
    try:
        # loop over all epochs
        for epoch in tqdm(range(config.data.train.num_epochs), desc="epoch"):

            # validate networks
            dataset.eval()
            for instance in training_instances.values():
                instance.network.eval()
                instance.network.clearState()  # reset last states
                instance.criterion.eval()  # enter validation mode for loss printing

            with torch.no_grad():
                # loop over samples
                for sample_idx in tqdm(range(len(dataset)), desc="validation", leave=False):
                    images, poses, dynamics, segmentations, structure = utils.castDataTuple(dataset[sample_idx], tensor_type, unsqueeze=True)
                    visualization_samples = [] if sample_idx in validation_visualization_indices else None
                    # segmentation network
                    if config.data.train.segmentation_net.enable:
                        instance = training_instances['segmentation']
                        segmentation_output = instance.network(poses, dynamics)
                        loss = instance.criterion(segmentation_output, segmentations)
                        if visualization_samples is not None:
                            visualization_samples.append(utils.visualizeSegmentationMap(torch.argmax(segmentation_output[0], dim=0, keepdim=True), config.data.dataset.num_segmentation_labels))
                    # structure network
                    if config.data.train.structure_net.enable:
                        instance = training_instances['structure']
                        structure_output = instance.network(poses, dynamics, segmentations)
                        loss = instance.criterion(structure_output, structure, segmentations)
                        if visualization_samples is not None:
                            visualization_samples.append(utils.visualizeStructureMap(structure_output[0], tensor_type))
                    # render network
                    if config.data.train.render_net.enable:
                        instance = training_instances['render']
                        foreground_output, final_output = instance.network(poses, segmentations, structure, background)
                        loss = instance.criterion(foreground_output, final_output, images, segmentations)
                        if visualization_samples is not None:
                            visualization_samples.append(foreground_output[0, [2, 1, 0]])
                            visualization_samples.append(final_output[0, [2, 1, 0]])

                    if visualization_samples is not None:
                        dataset.visualizeSample(writer, 'validation', epoch, visualization_samples)

            # train networks
            dataset.train()
            for instance in training_instances.values():
                instance.network.train()
                instance.network.clearState()  # reset last states
                instance.criterion.train()
            # loop over samples
            for sample_idx in tqdm(range(len(dataset)), desc="training", leave=False):
                images, poses, dynamics, segmentations, structure = utils.castDataTuple(dataset[sample_idx], tensor_type, unsqueeze=True)
                visualization_samples = [] if sample_idx in training_visualization_indices else None
                # segmentation network
                if config.data.train.segmentation_net.enable:
                    instance = training_instances['segmentation']
                    instance.optimizer.zero_grad()
                    segmentation_output = instance.network(poses, dynamics, segmentations if sample_idx % 500 == 0 else None)
                    loss = instance.criterion(segmentation_output, segmentations)
                    loss.backward()
                    instance.optimizer.step()
                    instance.scheduler.step()
                    if visualization_samples is not None:
                        visualization_samples.append(utils.visualizeSegmentationMap(torch.argmax(segmentation_output[0], dim=0, keepdim=True), config.data.dataset.num_segmentation_labels))
                # structure network
                if config.data.train.structure_net.enable:
                    instance = training_instances['structure']
                    instance.optimizer.zero_grad()
                    structure_output = instance.network(poses, dynamics, segmentations, structure if sample_idx % 500 == 0 else None)
                    loss = instance.criterion(structure_output, structure, segmentations)
                    loss.backward()
                    instance.optimizer.step()
                    instance.scheduler.step()
                    if visualization_samples is not None:
                        visualization_samples.append(utils.visualizeStructureMap(structure_output[0], tensor_type))
                # render network
                if config.data.train.render_net.enable:
                    instance = training_instances['render']
                    instance.optimizer.zero_grad()
                    foreground_output, final_output = instance.network(poses, segmentations, structure, background, images if sample_idx % 500 == 0 else None)
                    loss = instance.criterion(foreground_output, final_output, images, segmentations)
                    loss.backward()
                    instance.optimizer.step()
                    instance.scheduler.step()
                    if visualization_samples is not None:
                        visualization_samples.append(foreground_output[0, [2, 1, 0]])
                        visualization_samples.append(final_output[0, [2, 1, 0]])

                if visualization_samples is not None:
                    dataset.visualizeSample(writer, 'training', epoch, visualization_samples)

            # print losses and save checkpoint
            for name, instance in training_instances.items():
                instance.criterion.printLoss(writer, epoch)
                if epoch % config.data.train.checkpoint_backup_step == 0:
                    utils.save_checkpoint(instance.network, instance.optimizer, os.path.join(checkpoint_output_path, '{0}_epoch_{1:03d}'.format(name, epoch)))

    except KeyboardInterrupt:
        # save checkpoint before closing
        utils.ColorLogger.print('keyboard interrupt. saving last checkpoint.', 'WARNING')
        for name, instance in training_instances.items():
            utils.save_checkpoint(instance.network, instance.optimizer, os.path.join(checkpoint_output_path, '{0}_manual_interrupt'.format(name)))

    writer.close()
    # save final networks
    for name, instance in training_instances.items():
        utils.save_checkpoint(instance.network, instance.optimizer, os.path.join(checkpoint_output_path, '{0}_final'.format(name)))


if __name__ == '__main__':
    main()
