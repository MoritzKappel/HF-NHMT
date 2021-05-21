'''Datasets.py: PyTorch dataset implementation'''

import sys
import os
import cv2
import numpy as np
import scipy.sparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

import utils
import config

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'


# dataset class storing an actor performance
class ActorDataset(Dataset):

    # constructor
    def __init__(self, reenactment_mode=False):
        super(ActorDataset, self).__init__()
        # set reenactment mode (only return images and skeletons/dynamics as no training data is available)
        self.reenactment_mode = reenactment_mode
        # set data paths
        self.image_path = os.path.join(config.data.dataset.path, 'images')
        self.segmentation_path = os.path.join(config.data.dataset.path, 'segmentation')
        self.edge_path = os.path.join(config.data.dataset.path, 'structure')
        self.skeleton_path = os.path.join(config.data.dataset.path, 'target_actors', config.data.dataset.target_actor_name, 'skeletons')
        self.dynamics_path = os.path.join(config.data.dataset.path, 'target_actors', config.data.dataset.target_actor_name, 'dynamics')
        # read filenames
        self.image_filenames = utils.list_sorted_files(self.image_path)
        self.skeleton_filenames = utils.list_sorted_files(self.skeleton_path)
        self.dynamics_filenames = utils.list_sorted_files(self.dynamics_path)
        if not self.reenactment_mode:
            self.segmentation_filenames = utils.list_sorted_files(self.segmentation_path)
            self.edge_filenames = utils.list_sorted_files(self.edge_path)
        # if not load dynamic, load the entire dataset into RAM
        if not config.data.dataset.load_dynamic:
            self.preloaded_images = []
            self.preloaded_segmentations = []
            self.preloaded_edges = []
            self.preloaded_skeletons = []
            self.preloaded_dynamics = []
            utils.ColorLogger.print('Loading dataset into RAM...', 'BOLD')
            for index in tqdm(range(len(self.image_filenames)), desc="loading dataset", leave=False):
                self.preloaded_images.append(cv2.imread(os.path.join(self.image_path, self.image_filenames[index])))
                self.preloaded_skeletons.append(scipy.sparse.load_npz(os.path.join(self.skeleton_path, self.skeleton_filenames[index])).tocoo())
                self.preloaded_dynamics.append(scipy.sparse.load_npz(os.path.join(self.dynamics_path, self.dynamics_filenames[index])).tocoo())
                if not self.reenactment_mode:
                    self.preloaded_segmentations.append(np.load(os.path.join(self.segmentation_path, self.segmentation_filenames[index])))
                    self.preloaded_edges.append(np.load(os.path.join(self.edge_path, self.edge_filenames[index])))
            utils.ColorLogger.print('Done!', 'BOLD')
        # perform training/validation split
        self.total_length = len(self.image_filenames)
        self.training_length = int((config.data.dataset.training_split * self.total_length) + 0.5)
        self.validation_length = self.total_length - self.training_length
        self.train()

    # set training mode (return data from trainingset)
    def train(self):
        self.training_mode = True

    # set eval mode (return data from validation set cut from the end of the sequence accoring to percentage defined in config)
    def eval(self):
        self.training_mode = False

    # loads a sample (image, pose, dynamics, segmentation labels, structure outlines)
    def loadSample(self, index):
        # load sample
        img = cv2.imread(os.path.join(self.image_path, self.image_filenames[index])) if config.data.dataset.load_dynamic else self.preloaded_images[index]
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        skeleton = np.asarray((scipy.sparse.load_npz(os.path.join(self.skeleton_path, self.skeleton_filenames[index])) if config.data.dataset.load_dynamic else self.preloaded_skeletons[index]).todense()).reshape((config.data.dataset.num_bone_channels, config.data.dataset.target_actor_height, config.data.dataset.target_actor_width))
        dynamics = np.asarray((scipy.sparse.load_npz(os.path.join(self.dynamics_path, self.dynamics_filenames[index])) if config.data.dataset.load_dynamic else self.preloaded_dynamics[index]).todense()).reshape((4 * config.data.dataset.num_bone_channels, config.data.dataset.target_actor_height, config.data.dataset.target_actor_width))
        edges = (np.load(os.path.join(self.edge_path, self.edge_filenames[index])) if config.data.dataset.load_dynamic else self.preloaded_edges[index]) if not self.reenactment_mode else None
        segmentation = (np.load(os.path.join(self.segmentation_path, self.segmentation_filenames[index])) if config.data.dataset.load_dynamic else self.preloaded_segmentations[index]) if not self.reenactment_mode else None
        if not self.reenactment_mode:
            clothes_mask = segmentation < 0
            for label in config.data.dataset.segmentation_clothes_labels:
                clothes_mask = np.logical_or(clothes_mask, segmentation == label)
            edges = clothes_mask * edges
        # return sample
        return (img, skeleton, dynamics, segmentation, edges)

    # load a static background image and fit to target actor size
    def getBackground(self, path=None):
        background = cv2.imread(config.data.dataset.background_path)
        if background is None:
            utils.ColorLogger.print('Dataset: invalid background image: {0}'.format(config.data.dataset.background_path), 'ERROR')
            sys.exit(0)
        if background.shape[0] != config.data.dataset.target_actor_height or background.shape[1] != config.data.dataset.target_actor_width:
            background = cv2.resize(background, (config.data.dataset.target_actor_width, config.data.dataset.target_actor_height))
        background = (torch.from_numpy(background).float() / 255.0).permute(2, 0, 1).unsqueeze(0)
        return background

    def __getitem__(self, idx):
        if not self.training_mode:
            idx = idx + self.training_length
        # return sample, use preloaded if dynamic is False
        sample = self.loadSample(idx)
        return [torch.from_numpy(x) if x is not None else x for x in sample]

    def __len__(self):
        return self.training_length if self.training_mode else self.validation_length

    # visualize list of images (1 or 3 channels with no batch dim) in tensorboard
    def visualizeSample(self, writer, tag, epoch, images):
        writer.add_images(tag, torch.stack([i.repeat(3, 1, 1) if i.shape[0] == 1 else i for i in images], dim=0), epoch, dataformats='NCHW')
