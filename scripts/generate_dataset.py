#! /usr/bin/env python3

'''generate_dataset.py: preprocesses data for data and motion retargeting.'''

import projectpath

import sys
import os
import argparse
import numpy as np
import numpy.ma as npma
import scipy.sparse
from tqdm import tqdm
import math
import json
import cv2
from pathlib import Path
from munch import DefaultMunch

with projectpath.context():
    import utils
    import config

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'

# GLOBALS - customize paths and parameters here if necessary
OPENPOSE_PATH = Path(__file__).resolve().parents[1] / 'thirdparty' / 'openpose'
SCHP_PATH = Path(__file__).resolve().parents[1] / 'thirdparty' / 'Self-Correction-Human-Parsing'
SCHP_CHECKPOINT = SCHP_PATH / 'exp-schp-201908301523-atr.pth'
MEDIAN_FILTER_KERNEL_SIZE = 7
DIFFERENTIAL_QUOTIENT_STEP = 1
KEYPOINT_CONFIDENCE_THRESHOLD = 0.4
POSE_NOSE_INDEX = 0
POSE_L_ANKLE_INDEX = 14
POSE_R_ANKLE_INDEX = 11
GPU_INDEX = 0
CREATE_VIDEOS = True
VIDEO_FRAMERATE = 50


# calculate segmentation masks
def generateSegmentations(input_dir):
    # setup io-paths
    image_path = os.path.join(input_dir, 'images/')
    segmentation_path = os.path.join(input_dir, 'segmentation/')
    # check if data already exists
    if os.path.exists(segmentation_path):
        utils.ColorLogger.print('segmentation directory already exists', 'BOLD')
        return
    # create output directory
    os.makedirs(segmentation_path, exist_ok=True)
    # check if checkpoint exists
    if not os.path.isfile(SCHP_CHECKPOINT):
        utils.ColorLogger.print('checkpoint not found: "{0}"'.format(SCHP_CHECKPOINT), 'ERROR')
        utils.ColorLogger.print('Download checkpoint from "https://drive.google.com/u/0/uc?export=download&confirm=Lzpf&id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP"', 'BOLD')
        sys.exit(0)
    # calculate segmentations
    command_string = "cd {0} && python simple_extractor.py --gpu {1} --logits --dataset atr --model-restore {2} --input-dir {3} --output-dir {4}".format(SCHP_PATH, GPU_INDEX, SCHP_CHECKPOINT, image_path, segmentation_path)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to run segmentation network using: "{0}"'.format(command_string), 'ERROR')
        sys.exit(0)
    # create videos from renderings
    if CREATE_VIDEOS:
        command_string = "cat {0} | ffmpeg -framerate {1} -i - -c:v libx264 -pix_fmt yuv420p {2}".format(os.path.join(segmentation_path, '*.png'), VIDEO_FRAMERATE, os.path.join(input_dir, 'segmentation_labels.mp4'))
        if os.system(command_string) != 0:
            utils.ColorLogger.print('failed to generate segmentation label video using: "{0}"'.format(command_string), 'ERROR')
    # remove visualization images to reduce memory consumption
    command_string = "cd {0} && rm *.png".format(segmentation_path)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to remove surplus body part output using: "{0}"'.format(command_string), 'ERROR')
    # reduce features to single label
    files = utils.list_sorted_files(segmentation_path)
    for file_index in tqdm(range(len(files)), desc="parsing segmentation maps", leave=False):
        file_path = os.path.join(segmentation_path, files[file_index])
        tensor = np.expand_dims(np.argmax(np.load(file_path), axis=2), axis=0).astype(np.uint8)
        np.save(file_path, tensor)
    return


# calculate 2d pose keypoints using openpose and temporal median filtering
def generatePoses(input_dir):
    # setup io-paths
    image_path = os.path.join(input_dir, 'images/')
    output_pose_path = os.path.join(input_dir, 'pose/')
    # check if data already exists
    if os.path.exists(output_pose_path):
        utils.ColorLogger.print('pose directory already exists', 'BOLD')
        return
    # create output directories
    os.makedirs(output_pose_path, exist_ok=True)

    # helper function to load and parse openpose keypoint detections
    def load_keypoints(filepath):
        try:
            # open file
            with open(filepath, mode='rb') as rawfile:
                data = json.load(rawfile)
                # if person is visible
                if len(data['people']) > 0:
                    # parse json to numpy
                    keypoints = np.array(data['people'][0]['pose_keypoints_2d']+data['people'][0]['hand_left_keypoints_2d']+data['people'][0]['hand_right_keypoints_2d']+data['people'][0]['face_keypoints_2d'], np.float32).reshape((-1, 3))
                    # threshold confidence to binary mask
                    keypoints[:, 2] = np.where(keypoints[:, 2] > KEYPOINT_CONFIDENCE_THRESHOLD, 1.0, 0.0)
                else:
                    # return zeros if no person was detected
                    keypoints = np.zeros((137, 3), np.float32)
                rawfile.close()
            return keypoints
        except EnvironmentError:
            utils.ColorLogger.print('failed to process keypoint file: {0}'.format(os.path.join(filepath)), 'ERROR')
            sys.exit(0)

    # run openpose
    command_string = "cd {0} && ./build/examples/openpose/openpose.bin --image_dir {1} --number_people_max 1 --face --hand --write_json {2} --display 0 --render_pose 2 --num_gpu 1 --num_gpu_start {3} --write_images {2}".format(OPENPOSE_PATH, image_path, output_pose_path, GPU_INDEX)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to run openpose using: "{0}"'.format(command_string), 'ERROR')
        sys.exit(0)
    # create videos from rendered frames
    if CREATE_VIDEOS:
        command_string = "cat {0} | ffmpeg -framerate {1} -i - -c:v libx264 -pix_fmt yuv420p {2}".format(os.path.join(output_pose_path, '*.png'), VIDEO_FRAMERATE, os.path.join(input_dir, 'skeleton_labels.mp4'))
        if os.system(command_string) != 0:
            utils.ColorLogger.print('failed to generate skeleton label video using: "{0}"'.format(command_string), 'ERROR')
    # remove  openpose skeleton renderings to reduce memory consumption
    command_string = "cd {0} && rm *.png".format(output_pose_path)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to remove openpose renderings using: "{0}"'.format(command_string), 'ERROR')
    # fetch list of input images (for naming)
    image_files = utils.list_sorted_files(image_path)
    # fetch list of openpose_output files
    pose_files = utils.list_sorted_files(output_pose_path)
    # load poses and create masked array for numpy filtering
    pose_keypoints_masked = []
    pose_confidence_masks = []
    for filename in pose_files:
        pose_data = load_keypoints(os.path.join(output_pose_path, filename))
        pose_keypoints_masked.append(npma.masked_array(pose_data[:, :2], mask=np.repeat(np.expand_dims(pose_data[:, 2] < 0.5, axis=1), repeats=2, axis=1)))
        pose_confidence_masks.append(np.expand_dims(pose_data[:, 2], axis=1))
    # create pose images
    for file_index in tqdm(range(len(pose_files)), desc="filtering 2d pose keypoint tensors", leave=False):
        # median filter keypoints (only include neighborhood points over confidence threshold)
        filtered_keypoints = npma.median(npma.stack([pose_keypoints_masked[min(len(pose_files) - 1, max(0, file_index + i))] for i in range(-MEDIAN_FILTER_KERNEL_SIZE // 2, MEDIAN_FILTER_KERNEL_SIZE // 2)], axis=0), axis=0).astype(np.float32)
        # save filtered tensor (fuse with original confidence mask)
        filename = image_files[file_index].split('.')[0]
        np.save(os.path.join(output_pose_path, filename), np.concatenate([filtered_keypoints.filled(fill_value=0.0), pose_confidence_masks[file_index]], axis=1))
    # remove original openpose keypoints to reduce memory consumption
    command_string = "cd {0} && rm *.json".format(output_pose_path)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to remove openpose keypoints using: "{0}"'.format(command_string), 'ERROR')
    return


# creates skeleton and dynamics rasterizations for training or given target actor
def generateSkeletons(input_dir, target_actor=None):
    # set mode
    pose_norm_required = target_actor is not None
    # setup io-paths
    target_actor_prefix = 'self' if not pose_norm_required else os.path.basename(os.path.normpath(target_actor))
    target_actor_path = os.path.join(input_dir, 'target_actors', target_actor_prefix)
    skeleton_path = os.path.join(target_actor_path, 'skeletons')
    dynamics_path = os.path.join(target_actor_path, 'dynamics')
    # check if data already exists
    if os.path.exists(target_actor_path):
        utils.ColorLogger.print('target actor inputs already exist', 'BOLD')
        return
    # create output directories
    os.makedirs(skeleton_path, exist_ok=True)
    os.makedirs(dynamics_path, exist_ok=True)

    # calculate dynamics scaling and pose normalization parameters over datasets
    pose_path_source = os.path.join(input_dir, 'pose')
    pose_files_source = utils.list_sorted_files(pose_path_source)
    pose_path_target = pose_path_source if not pose_norm_required else os.path.join(target_actor, 'pose')
    pose_files_target = utils.list_sorted_files(pose_path_target)
    image_path_target = os.path.join(input_dir if not pose_norm_required else target_actor, 'images')
    image_files_target = utils.list_sorted_files(image_path_target)
    image_path_source = os.path.join(input_dir, 'images')
    image_files_source = utils.list_sorted_files(image_path_source)
    height_target, width_target, channels_target = cv2.imread(os.path.join(image_path_target, image_files_target[0])).shape
    velocity_factor = 0.0
    acceleration_factor = 0.0
    target_heights = []
    target_scales = []
    # extract statistics from target dataset (equals source actor for training data generation)
    for index in tqdm(range(len(pose_files_target)), desc="gathering target actor statistics...", leave=False):
        # load poses
        last_keypoints = np.load(os.path.join(pose_path_target, pose_files_target[max(0, index - DIFFERENTIAL_QUOTIENT_STEP)]))
        current_keypoints = np.load(os.path.join(pose_path_target, pose_files_target[index]))
        next_keypoints = np.load(os.path.join(pose_path_target, pose_files_target[min(len(pose_files_target)-1, index + DIFFERENTIAL_QUOTIENT_STEP)]))
        combined_mask = np.expand_dims(last_keypoints[:, 2] * current_keypoints[:, 2] * next_keypoints[:, 2], axis=1)
        # calc dynamics (finite elements)
        keypoint_velocities = ((current_keypoints[:, :2] - last_keypoints[:, :2]) / DIFFERENTIAL_QUOTIENT_STEP) * combined_mask
        keypoint_accelerations = ((next_keypoints[:, :2] + last_keypoints[:, :2] - (2.0 * current_keypoints[:, :2])) / DIFFERENTIAL_QUOTIENT_STEP) * combined_mask
        velocities_norm = np.linalg.norm(keypoint_velocities, ord=2, axis=1)
        accelerations_norm = np.linalg.norm(keypoint_accelerations, ord=2, axis=1)
        if np.count_nonzero(velocities_norm):
            velocity_factor += (np.sum(velocities_norm) / np.count_nonzero(velocities_norm))
        if np.count_nonzero(accelerations_norm):
            acceleration_factor += (np.sum(accelerations_norm) / np.count_nonzero(accelerations_norm))
        # gather data for pose normalization
        if pose_norm_required:
            keypoint_ankle_left = current_keypoints[POSE_L_ANKLE_INDEX]
            keypoint_ankle_right = current_keypoints[POSE_R_ANKLE_INDEX]
            keypoint_nose = current_keypoints[POSE_NOSE_INDEX]
            if keypoint_ankle_left[2] * keypoint_ankle_right[2] * keypoint_nose[2] > 0.0:
                # mean_ankles = (keypoint_ankle_left[:2] + keypoint_ankle_right[:2]) / 2.0
                mean_ankles = np.array([(keypoint_ankle_left[0] + keypoint_ankle_right[0]) / 2.0, max(keypoint_ankle_left[1], keypoint_ankle_right[1])], np.float32)
                target_heights.append(mean_ankles)
                target_scales.append(np.linalg.norm(keypoint_nose[:2] - mean_ankles, ord=2, axis=0))
    velocity_factor = len(pose_files_target) / velocity_factor
    acceleration_factor = len(pose_files_target) / acceleration_factor

    # extract statistics from source dataset
    if pose_norm_required:  # not needed for training data
        source_heights = []
        source_scales = []
        height_source, width_source, channels_source = cv2.imread(os.path.join(image_path_source, image_files_source[0])).shape
        for index in tqdm(range(len(pose_files_source)), desc="gathering source actor statistics...", leave=False):
            # load poses
            current_keypoints = np.load(os.path.join(pose_path_source, pose_files_source[index]))
            # gather data for pose normalization
            keypoint_ankle_left = current_keypoints[POSE_L_ANKLE_INDEX]
            keypoint_ankle_right = current_keypoints[POSE_R_ANKLE_INDEX]
            keypoint_nose = current_keypoints[POSE_NOSE_INDEX]
            if keypoint_ankle_left[2] * keypoint_ankle_right[2] * keypoint_nose[2] > 0.0:
                # mean_ankles = (keypoint_ankle_left[:2] + keypoint_ankle_right[:2]) / 2.0
                mean_ankles = np.array([(keypoint_ankle_left[0] + keypoint_ankle_right[0]) / 2.0, max(keypoint_ankle_left[1], keypoint_ankle_right[1])], np.float32)
                source_heights.append(mean_ankles)
                source_scales.append(np.linalg.norm(keypoint_nose[:2] - mean_ankles, ord=2, axis=0))
        # calculate final pose normalization parameters
        target_heights = np.array(target_heights, np.float32)
        target_scales = np.array(target_scales, np.float32)
        source_heights = np.array(source_heights, np.float32)
        source_scales = np.array(source_scales, np.float32)
        s_close = np.max(source_heights, axis=0)[1]
        s_far = 2 * np.median(source_heights, axis=0)[1] - s_close
        t_close = np.max(target_heights, axis=0)[1]
        t_far = 2 * np.median(target_heights, axis=0)[1] - t_close
        h_s_close = source_scales[np.argmax(source_heights[1], axis=0)]
        h_s_far = 2 * np.median(source_scales) - h_s_close
        h_t_close = target_scales[np.argmax(target_heights[1], axis=0)]
        h_t_far = 2 * np.median(target_scales) - h_t_close
        c_close = h_t_close / h_s_close
        c_far = h_t_far / h_s_far
        s_horizontal = np.mean(source_heights, axis=0)[0]
        t_horizontal = np.mean(target_heights, axis=0)[0]
        # print(s_close, s_far, t_close, t_far, h_s_close, h_s_far, h_t_close, h_t_far, c_close, c_far, s_horizontal, t_horizontal, velocity_factor, acceleration_factor)

    # draw skeletons and dynamics and store as sparse matrices
    pose_skeleton_index_array = [[[0, 1], [1, 8]], [[0, 15], [15, 17], [0, 16], [16, 18]], [[1, 5], [5, 6], [6, 7]], [[1, 2], [2, 3], [3, 4]], [[8, 12], [12, 13], [13, 14]], [[8, 9], [9, 10], [10, 11]], [[14, 21], [14, 19], [19, 20]], [[11, 24], [11, 22], [22, 23]], [[25, 26], [26, 27], [27, 28], [28, 29], [25, 30], [30, 31], [31, 32], [32, 33], [25, 34], [34, 35], [35, 36], [36, 37], [25, 38], [38, 39], [39, 40], [40, 41], [25, 42], [42, 43], [43, 44], [44, 45]], [[46, 47], [47, 48], [48, 49], [49, 50], [46, 51], [51, 52], [52, 53], [53, 54], [46, 55], [55, 56], [56, 57], [57, 58], [46, 59], [59, 60], [60, 61], [61, 62], [46, 63], [63, 64], [64, 65], [65, 66]], [[67, 68], [68, 69], [69, 70], [70, 71], [71, 72], [72, 73], [73, 74], [75, 76], [76, 77], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [82, 83], [84, 85], [85, 86], [86, 87], [87, 88], [89, 90], [90, 91], [91, 92], [92, 93], [94, 95], [95, 96], [96, 97], [98, 99], [99, 100], [100, 101], [101, 102], [103, 104], [104, 105], [105, 106], [106, 107], [107, 108], [108, 109], [109, 110], [110, 111], [111, 112], [112, 113], [113, 114], [114, 109], [115, 116], [116, 117], [117, 118], [118, 119], [119, 120], [120, 121], [121, 122], [122, 123], [123, 124], [124, 125], [125, 126], [126, 115], [127, 128], [128, 129], [129, 130], [130, 131], [131, 132], [132, 133], [133, 134], [134, 127], [104, 135], [135, 105], [110, 136], [136, 111]]]
    for index in tqdm(range(len(pose_files_source)), desc="drawing skeleton and dynamics tensors", leave=False):
        # load poses and dynamics
        last_keypoints = np.load(os.path.join(pose_path_source, pose_files_source[max(0, index - DIFFERENTIAL_QUOTIENT_STEP)]))
        current_keypoints = np.load(os.path.join(pose_path_source, pose_files_source[index]))
        next_keypoints = np.load(os.path.join(pose_path_source, pose_files_source[min(len(pose_files_source)-1, index + DIFFERENTIAL_QUOTIENT_STEP)]))
        combined_mask = np.expand_dims(last_keypoints[:, 2] * current_keypoints[:, 2] * next_keypoints[:, 2], axis=1)
        # normalize pose for given target actor:
        if pose_norm_required:
            # apply transformation
            for kps in (last_keypoints, current_keypoints, next_keypoints):
                keypoint_ankle_left = kps[POSE_L_ANKLE_INDEX]
                keypoint_ankle_right = kps[POSE_R_ANKLE_INDEX]
                current__pos = np.array([s_horizontal, max(keypoint_ankle_left[1], keypoint_ankle_right[1])], np.float32) if keypoint_ankle_left[2] * keypoint_ankle_right[2] > 0.0 else np.array([s_horizontal, s_far], np.float32)
                final_scale = c_far + ((current__pos[1] - s_far) / (s_close - s_far)) * (c_close - c_far)
                final_translation = np.array([t_horizontal, t_far + ((current__pos[1] - s_far) / (s_close - s_far)) * (t_close - t_far)], np.float32)
                kps[:, :2] = ((kps[:, :2] - current__pos) * final_scale) + final_translation
        # calc dynamics (finite elements)
        keypoint_velocities = ((current_keypoints[:, :2] - last_keypoints[:, :2]) / DIFFERENTIAL_QUOTIENT_STEP) * combined_mask * velocity_factor
        keypoint_accelerations = ((next_keypoints[:, :2] + last_keypoints[:, :2] - (2.0 * current_keypoints[:, :2])) / DIFFERENTIAL_QUOTIENT_STEP) * combined_mask * acceleration_factor
        # draw per bone pose and dynamics
        pose_images = []
        dynamics_images = []
        for limb_index in range(len(pose_skeleton_index_array)):
            limb_pose_image = np.zeros((height_target, width_target), np.float32)  # single channel: binary skeleton
            limb_dynamics_image = np.zeros((4, height_target, width_target), np.float32)  # 4 channels: velocity x,y and acceleration x,y

            # draw binary skeleton
            for start_index, end_index in pose_skeleton_index_array[limb_index]:
                bone_pose_image = np.zeros((height_target, width_target), np.float32)
                bone_dynamics_image = np.zeros((4, height_target, width_target), np.float32)
                start = (int((current_keypoints[start_index, 0]) + 0.5), int((current_keypoints[start_index, 1]) + 0.5))
                end = (int((current_keypoints[end_index, 0]) + 0.5), int((current_keypoints[end_index, 1]) + 0.5))
                draw_bone = current_keypoints[start_index, 2] > 0.0 and current_keypoints[end_index, 2] > 0.0
                # check if bone was detected
                if draw_bone:
                    # draw bone
                    cv2.line(bone_pose_image, start, end, color=1.0, thickness=1, lineType=cv2.LINE_8)
                    # interpolate keypoint dynamics along bone
                    coords_h, coords_w = np.where((bone_pose_image > 0.0))
                    values_p1 = np.concatenate([keypoint_velocities[start_index], keypoint_accelerations[start_index]], axis=0)
                    values_p2 = np.concatenate([keypoint_velocities[end_index], keypoint_accelerations[end_index]], axis=0)
                    dist_p1 = np.sqrt(((coords_w - current_keypoints[start_index, 0])**2) + ((coords_h - current_keypoints[start_index, 1])**2))
                    dist_p2 = np.sqrt(((coords_w - current_keypoints[end_index, 0])**2) + ((coords_h - current_keypoints[end_index, 1])**2))
                    interp_values = (np.outer(values_p1, dist_p2) + np.outer(values_p2, dist_p1)) / (dist_p1 + dist_p2)
                    bone_dynamics_image[:, coords_h, coords_w] = interp_values
                # accumulate bones
                limb_pose_image += bone_pose_image
                limb_dynamics_image += bone_dynamics_image
            # append to outputs
            limb_dynamics_image = np.divide(limb_dynamics_image, limb_pose_image, out=np.zeros_like(limb_dynamics_image), where=limb_pose_image > 0.0)
            limb_pose_image = np.clip(limb_pose_image, 0.0, 1.0)
            dynamics_images.append(limb_dynamics_image)
            pose_images.append(limb_pose_image)
        # save outputs
        final_pose = np.stack(pose_images, axis=0).astype(np.float32)
        final_dynamics = np.concatenate(dynamics_images, axis=0).astype(np.float32)
        filename = image_files_source[index].split('.')[0]
        # write skeleton visualization
        cv2.imwrite(os.path.join(skeleton_path, '{0}_pose.png'.format(filename)), (np.sum(final_pose, axis=0) * 255).astype(np.uint8))
        # write dynamics visualization (velocity only)
        cv2.imwrite(os.path.join(dynamics_path, '{0}_velocity.png'.format(filename)), (utils.pseudocolorOpticalFlowNumpy(np.sum(np.stack([final_dynamics[[(j * 4) + 0, (j * 4) + 1]] for j in range(11)], axis=0), axis=0), norm_factor=1.0) * 255.0).astype(np.uint8))
        # write training tensors as sparse matrices
        scipy.sparse.save_npz(os.path.join(skeleton_path, filename), scipy.sparse.coo_matrix(final_pose.flatten()))
        scipy.sparse.save_npz(os.path.join(dynamics_path, filename), scipy.sparse.coo_matrix(final_dynamics.flatten()))
    # create videos from rendered skeleton
    if CREATE_VIDEOS:
        command_string = "cd {0} && ffmpeg -pattern_type glob -framerate {2} -i {1} -c:v libx264 -pix_fmt yuv420p {3}".format(skeleton_path, "'*.png'", VIDEO_FRAMERATE, os.path.join(target_actor_path, 'pose_conditioning.mp4'))
        if os.system(command_string) != 0:
            utils.ColorLogger.print('failed to generate pose conditioning video using: "{0}"'.format(command_string), 'ERROR')
        command_string = "cd {0} && ffmpeg -pattern_type glob -framerate {2} -i {1} -c:v libx264 -pix_fmt yuv420p {3}".format(dynamics_path, "'*.png'", VIDEO_FRAMERATE, os.path.join(target_actor_path, 'dynamics_conditioning.mp4'))
        if os.system(command_string) != 0:
            utils.ColorLogger.print('failed to generate dynamics conditioning video using: "{0}"'.format(command_string), 'ERROR')
    # remove skeleton and velocity images
    command_string = "cd {0} && rm *.png".format(skeleton_path)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to remove skeleton images using: "{0}"'.format(command_string), 'ERROR')
    command_string = "cd {0} && rm *.png".format(dynamics_path)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to remove velocity images using: "{0}"'.format(command_string), 'ERROR')


# generates structure images using gabor filter
def generateStructure(input_dir):
    # setup io-paths
    image_path = os.path.join(input_dir, 'images/')
    structure_path = os.path.join(input_dir, 'structure/')
    # check if data already exists
    if os.path.exists(structure_path):
        utils.ColorLogger.print('structure directory already exists', 'BOLD')
        return
    # create output directory
    os.makedirs(structure_path, exist_ok=True)
    # manual parameter adjustment
    utils.ColorLogger.print('confirm parameter selection using the ESC key', 'BOLD')
    gr = GaborRenderer(image_path)
    gr.run()
    # fetch list of input images (for naming)
    image_files = utils.list_sorted_files(image_path)
    # outputs
    for index in tqdm(range(len(image_files)), desc="generating structure images", leave=False):
        gr.changeImage(index)
        final_frame = gr.generateFinalFrame()
        final_frame_rgb = (cv2.cvtColor(np.concatenate((final_frame, np.ones((1, final_frame.shape[1], final_frame.shape[2]), dtype=final_frame.dtype)), axis=0).transpose((1, 2, 0)), cv2.COLOR_HSV2RGB) * 255.0).astype(np.uint8)
        final_frame[0] /= 360.0
        # save result
        filename = image_files[index].split('.')[0]
        np.save(os.path.join(structure_path, filename), final_frame)
        cv2.imwrite(os.path.join(structure_path, filename + '_vis.png'), final_frame_rgb)
    # create videos from rendered structure
    if CREATE_VIDEOS:
        command_string = "cat {0} | ffmpeg -framerate {1} -i - -c:v libx264 -pix_fmt yuv420p {2}".format(os.path.join(structure_path, '*.png'), VIDEO_FRAMERATE, os.path.join(input_dir, 'structure_labels.mp4'))
        if os.system(command_string) != 0:
            utils.ColorLogger.print('failed to generate structure label video using: "{0}"'.format(command_string), 'ERROR')
    # remove visualization images to reduce memory consumption
    command_string = "cd {0} && rm *.png".format(structure_path)
    if os.system(command_string) != 0:
        utils.ColorLogger.print('failed to remove structure visualizations using: "{0}"'.format(command_string), 'ERROR')


# generates a new training config file for the given dataset
def generateConfig(input_dir):
    config_path = Path(__file__).resolve().parents[1] / 'configs' / (os.path.basename(os.path.normpath(input_dir)) + '.yaml')
    # see if config already exists
    if os.path.exists(config_path):
        utils.ColorLogger.print('dataset config file already exists', 'BOLD')
        return
    # get image shape
    image_path = os.path.join(input_dir, 'images')
    height, width, channels = cv2.imread(os.path.join(image_path, utils.list_sorted_files(image_path)[0])).shape
    # create new config for dataset
    config.data = DefaultMunch()
    # dataset parameters
    config.data.dataset = DefaultMunch()
    config.data.dataset.path = input_dir
    config.data.dataset.load_dynamic = True
    config.data.dataset.training_split = 0.95
    config.data.dataset.num_segmentation_labels = 18
    config.data.dataset.num_bone_channels = 11
    config.data.dataset.segmentation_clothes_labels = [1, 4, 5, 6, 7, 16, 17]
    config.data.dataset.segmentation_background_label = 0
    config.data.dataset.background_path = os.path.join(input_dir, 'background.png')
    config.data.dataset.target_actor_name = 'self'
    config.data.dataset.target_actor_width = width
    config.data.dataset.target_actor_height = height
    # training parameters
    config.data.train = DefaultMunch()
    config.data.train.name_prefix = os.path.basename(os.path.normpath(input_dir))
    config.data.train.tensorboard_logdir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 'tensorboard')
    config.data.train.output_checkpoints_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 'checkpoints')
    config.data.train.checkpoint_backup_step = 5
    config.data.train.training_visualization_indices = (0.2, 0.5, 0.8)
    config.data.train.validation_visualization_indices = (0.2, 0.4, 0.8)
    config.data.train.gpu_index = 0
    config.data.train.num_epochs = 50
    config.data.train.use_cuda = True
    config.data.train.vgg_layers = (1, 6)
    config.data.train.render_net = DefaultMunch()
    config.data.train.render_net.adam_beta1 = 0.5
    config.data.train.render_net.adam_beta2 = 0.999
    config.data.train.render_net.enable = True
    config.data.train.render_net.last_checkpoint = None
    config.data.train.render_net.learningrate = 0.0002
    config.data.train.render_net.learningrate_decay_factor = 0.5
    config.data.train.render_net.learningrate_decay_step = 300000
    config.data.train.render_net.loss_lambda_final_perceptive = 0.5
    config.data.train.render_net.loss_lambda_final_reconstruction = 0.5
    config.data.train.render_net.loss_lambda_foreground_perceptive = 0.9
    config.data.train.render_net.loss_lambda_foreground_reconstruction = 0.1
    config.data.train.segmentation_net = DefaultMunch()
    config.data.train.segmentation_net.adam_beta1 = 0.5
    config.data.train.segmentation_net.adam_beta2 = 0.999
    config.data.train.segmentation_net.enable = True
    config.data.train.segmentation_net.last_checkpoint = None
    config.data.train.segmentation_net.learningrate = 0.0002
    config.data.train.segmentation_net.learningrate_decay_factor = 0.5
    config.data.train.segmentation_net.learningrate_decay_step = 300000
    config.data.train.segmentation_net.loss_lambda = 1.0
    config.data.train.structure_net = DefaultMunch()
    config.data.train.structure_net.adam_beta1 = 0.5
    config.data.train.structure_net.adam_beta2 = 0.999
    config.data.train.structure_net.enable = True
    config.data.train.structure_net.last_checkpoint = None
    config.data.train.structure_net.learningrate = 0.0002
    config.data.train.structure_net.learningrate_decay_factor = 0.5
    config.data.train.structure_net.learningrate_decay_step = 300000
    config.data.train.structure_net.loss_lambda = 0.5
    # inference parameters
    config.data.infer = DefaultMunch()
    config.data.infer.use_cuda = True
    config.data.infer.gpu_index = 0
    config.data.infer.segmentation_checkpoint = None
    config.data.infer.structure_checkpoint = None
    config.data.infer.render_checkpoint = None
    config.data.infer.use_gt_segmentation = False
    config.data.infer.use_gt_structure = False
    config.data.infer.num_initial_iterations = 50
    config.data.infer.structure_magnification = 1.0
    config.data.infer.output_dir = None
    config.data.infer.append_source_image = False
    config.data.infer.generate_segmentations = True
    config.data.infer.generate_structure = True
    config.data.infer.validation_set = True
    config.data.infer.create_videos = True
    config.data.infer.video_framerate = 50
    # save config
    config.saveConfig(config_path)


# helper class for config parameter detection
class GaborRenderer():
    def __init__(self, input_images):
        # vars
        self.filter_size = 25
        self.setSigma(1)
        self.setTheta(0)
        self.setLambd(40)
        self.setGamma(30)
        self.setVectorSmoothing(5)
        self.setNormalization(20)

        self.image = None
        self.image_gray = None
        self.input_images = input_images
        self.image_filenames = utils.list_sorted_files(input_images)
        self.window_name = 'Gabor Filter'
        #
        self.changeImage(0)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.render_rate = 100

    def generateFinalFrame(self):
        # final result
        num_filter = 32
        discrete_kernels = [cv2.getGaborKernel((self.filter_size, self.filter_size), self.sigma, math.pi * i / num_filter, self.lambd, self.gamma, ktype=cv2.CV_32F) for i in range(num_filter)]
        activations = np.stack([np.absolute(cv2.filter2D(self.image_gray, -1, i)) for i in discrete_kernels], axis=0)
        confidence = np.expand_dims(np.amax(activations, axis=0) / cv2.GaussianBlur(self.image_gray, ksize=(7, 7), sigmaX=2.0), axis=2)
        orientation = np.argmax(activations, axis=0) * 2.0 * math.pi / num_filter
        vectors = np.stack((np.cos(orientation), np.sin(orientation)), axis=2)
        vectors_smoothed = cv2.GaussianBlur(vectors * confidence, ksize=(7, 7), sigmaX=self.vectorSmoothing).astype(np.float32)
        (x, y) = (np.array(vectors_smoothed[:, :, 0], copy=True), np.array(vectors_smoothed[:, :, 1], copy=True))
        (magnitude, angle) = cv2.cartToPolar(x, y, angleInDegrees=True)
        magnitude = np.divide(magnitude, self.normalization)
        ret, magnitude = cv2.threshold(magnitude, 1.0, 0, cv2.THRESH_TRUNC)
        return np.stack((angle, magnitude), axis=0)

    def showFrame(self):
        # get images
        g_kernel = cv2.getGaborKernel((self.filter_size, self.filter_size), self.sigma, self.theta, self.lambd, self.gamma, ktype=cv2.CV_32F)
        scale_size = min(self.image_gray.shape[0], self.image_gray.shape[1])
        x_padding = self.image_gray.shape[1] - scale_size
        y_padding = self.image_gray.shape[0] - scale_size
        g_kernel_color = cv2.copyMakeBorder(cv2.resize(cv2.applyColorMap(cv2.normalize(src=g_kernel, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1), cv2.COLORMAP_JET), (scale_size, scale_size)), top=y_padding//2, left=x_padding//2, bottom=y_padding//2, right=x_padding//2, borderType=cv2.BORDER_CONSTANT)
        filtered_img = cv2.applyColorMap(cv2.normalize(src=np.absolute(cv2.filter2D(self.image_gray, -1, g_kernel)), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1), cv2.COLORMAP_JET)
        final_frame = self.generateFinalFrame()
        final_frame_rgb = (cv2.cvtColor(np.concatenate((final_frame, np.ones((1, final_frame.shape[1], final_frame.shape[2]), dtype=final_frame.dtype)), axis=0).transpose((1, 2, 0)), cv2.COLOR_HSV2RGB) * 255.0).astype(np.uint8)
        # combine and print
        combined = np.concatenate((self.image, filtered_img, g_kernel_color, final_frame_rgb), axis=1)
        cv2.imshow(self.window_name, combined)

    def setSigma(self, val):
        self.sigma = val + 0.0001

    def setTheta(self, val):
        self.theta = val * math.pi / 100.0

    def setLambd(self, val):
        self.lambd = val + 0.0001

    def setGamma(self, val):
        self.gamma = (val / 100.0) + 0.0001

    def changeImage(self, idx):
        self.image = cv2.imread(os.path.join(self.input_images, self.image_filenames[idx]))
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def setVectorSmoothing(self, val):
        self.vectorSmoothing = (val / 10.0) + 0.0001

    def setNormalization(self, val):
        self.normalization = (val / 50) + 0.0001

    def run(self):
        # create window
        self.window = cv2.namedWindow(self.window_name)
        cv2.createTrackbar('image_index', self.window_name, 0, len(self.image_filenames), self.changeImage)
        cv2.createTrackbar('sigma', self.window_name, 1, 100, self.setSigma)
        cv2.createTrackbar('theta', self.window_name, 0, 100, self.setTheta)
        cv2.createTrackbar('lambd', self.window_name, 40, 100, self.setLambd)
        cv2.createTrackbar('gamma', self.window_name, 30, 100, self.setGamma)
        cv2.createTrackbar('vector smoothing', self.window_name, 5, 100, self.setVectorSmoothing)
        cv2.createTrackbar('normalization', self.window_name, 20, 100, self.setNormalization)
        # run mainloop
        key = 0
        while key != 27:
            self.showFrame()
            key = cv2.waitKey(self.render_rate)
        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)


# main func for argparsing
def main():
    # parse conmmand line arguments
    parser = argparse.ArgumentParser(description='Ppreprocesses required data for actor video sequence.')
    parser.add_argument('-i', '--input_dir', dest='input_dir', metavar='PATH/TO/SEQUENCE/DIRECTORY', required=True,
                        help='Root directory of the data sequence, which contains the "images/" subdirectory.')
    parser.add_argument('--pose_norm', dest='pose_norm', metavar='PATH/TO/TARGETSEQUENCE/DIRECTORY', required=False,
                        help='(optional) Directory of target actor for whom to generate reenactment inputs (dataset directory that was used for training). Target must contain image and pose annotations generated with --train flag!.')
    parser.add_argument('--train', dest='train_mode', action='store_true', help='create training data for this actor')
    args = parser.parse_args()

    # check if input dir is valid (image dir exists)
    utils.ColorLogger.print('validating input directory...', 'BOLD')
    if not os.path.exists(os.path.join(args.input_dir, 'images')):
        utils.ColorLogger.print('invalid input directory (no image sequence provided)!', 'ERROR')
        sys.exit(0)
    utils.ColorLogger.print('done', 'OKGREEN')

    if args.train_mode:
        utils.ColorLogger.print('generating structure images...', 'BOLD')
        generateStructure(args.input_dir)
        utils.ColorLogger.print('done', 'OKGREEN')
        utils.ColorLogger.print('calculating segmentation masks...', 'BOLD')
        generateSegmentations(args.input_dir)
        utils.ColorLogger.print('done', 'OKGREEN')
        utils.ColorLogger.print('calculating pose keypoints...', 'BOLD')
        generatePoses(args.input_dir)
        utils.ColorLogger.print('done', 'OKGREEN')
        utils.ColorLogger.print('generating skeleton and dynamics rasterizations...', 'BOLD')
        generateSkeletons(args.input_dir)
        utils.ColorLogger.print('done', 'OKGREEN')
        utils.ColorLogger.print('generating training config file...', 'BOLD')
        generateConfig(args.input_dir)
        utils.ColorLogger.print('done', 'OKGREEN')

    if args.pose_norm is not None:
        # check if input dir is valid (image dir exists)
        utils.ColorLogger.print('validating target actor directory for pose normalization...', 'BOLD')
        if not os.path.exists(os.path.join(args.pose_norm, 'images')) and os.path.exists(os.path.join(args.pose_norm, 'pose')):
            utils.ColorLogger.print('invalid target directory (no image and pose sequence provided)!', 'ERROR')
            sys.exit(0)
        # generate poses for source actor in case they do not already exist
        utils.ColorLogger.print('calculating pose keypoints...', 'BOLD')
        generatePoses(args.input_dir)
        # generate normalized poses for target actor
        utils.ColorLogger.print('generating skeleton and dynamics rasterizations...', 'BOLD')
        generateSkeletons(args.input_dir, target_actor=args.pose_norm)
        utils.ColorLogger.print('done', 'OKGREEN')


# run script
if __name__ == '__main__':
    main()
