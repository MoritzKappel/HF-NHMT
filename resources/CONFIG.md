# HF-NHMT - Configuration file

This project uses configuration files instead of command line arguments to simplify script execution.
A default config is provided in the *configs* directory.
Additionally, a new config file with some preconfigurations will automatically be created after preprocessing a new dataset.

The following documentation describes all configuration options. Please remember to modify the configuration before running the according scripts.

---
## Dataset parameters
### path
The system path of the dataset to use, containing the images, background, and annotations. During inference, this is the dataset containing the *source* actor annotations (poses).
### load_dynamic
If set to false, the dataset will be preloaded into RAM.
### training_split
Percentage of data (starting from the beginning) that will be used for training. The rest will be used as validation set.
For example, a value of 1.0 indicates that all samples are used for training, while a 0.8 will cut the last 20% of the video for validation/testing.
### num_segmentation_labels
Number of body part segmentation labels.We use the ATR dataset containing 18 unique labels.
### num_bone_channels
Number of channels (limbs) in the skeleton representation. Default is 11.
### segmentation_clothes_labels
List containing all segmentation labels refering to clothes.
Only labels in this list will be processed by the garmnent structure network.
### segmentation_background_label
Segmentation label for the scene background
### background_path
Path of the scene background. During inference, this should be the background used for training the requested render network component.
### target_actor_name
Directory name of the dataset used to train the executed network, if pose normalization is required.
To use the unadjusted skeletons (e.g. during training), this parameter should be set to *self*.
### target_actor_width
Image width of the *target* actor dataset (equals current dataset width during training).
### target_actor_height
Image height of the *target* actor dataset (equals current dataset width during training).

---
## Training parameters
### name_prefix
Name of the current training run, used for checkpoint and tensorboard log file naming.
### tensorboard_logdir
Output path for tensorboard logs. 
### output_checkpoints_dir
Putput path for training checkpoints.
### checkpoint_backup_step
Distance between training checkpoints (in epochs).
E.g. a value of 5 will generate a training checkpoint every 5 epochs. Use -1 to disable.
### training_visualization_indices
List of relative trainingset indices (in range [0, 1]) used for training progress visualization in tensorboard.
### validation_visualization_indices
List of relative validationset indices (in range [0, 1]) used for training progress visualization in tensorboard.
### gpu_index
Index of GPU to train on.
### num_epochs
Number of training epochs
### use_cuda
Activates GPU training
### vgg_layers
VGG layers used in perceptive loss
### <u>render_net</u>
Contains parameters for the rendering (appearance) network component.
#### adam_beta1
First beta parameter for ADAM optimization.
#### adam_beta2
Second beta parameter for ADAM optimization.
#### enable
Set to False to skip training the rendering (appearance) component.
#### last_checkpoint
Path of rendering network checkpoint to resume training from.
Use None to train from scratch.
#### learningrate
The optimizer learning rate.
#### learningrate_decay_factor
Decay factor applied to the learning rate.
#### learningrate_decay_step
Number of of optimization steps before learning rate decay is applied.
#### loss_lambda_final_perceptive
Perceptive (VGG) loss lambda for background fusion.
#### oss_lambda_final_reconstruction
Reconstruction (L1) loss lambda for background fusion.
#### loss_lambda_foreground_perceptive
Perceptive (VGG) loss lambda for actor generation.
#### loss_lambda_foreground_reconstruction
Reconstruction (L1) loss lambda for actor generation.
### <u>segmentation_net</u>
Contains parameters for the segmentation (shape) network component.
#### adam_beta1
First beta parameter for ADAM optimization.
#### adam_beta2
Second beta parameter for ADAM optimization.
#### enable
Set to False to skip training the segmentation (shape) component.
#### last_checkpoint
Path of segmentation network checkpoint to resume training from. Use None to train from scratch.
#### learningrate
The optimizer learning rate.
#### learningrate_decay_factor
Decay factor applied to the learning rate.
#### learningrate_decay_step
Number of of optimization steps before learning rate decay is applied.
#### loss_lambda
Binary cross entropy (BCE) lambda for segmentation learning.
### <u>structure_net</u>
Contains parameters for the structure network component.
#### adam_beta1
First beta parameter for ADAM optimization.
#### adam_beta2
Second beta parameter for ADAM optimization.
#### enable
Set to False to skip training the clothing structure component.
#### last_checkpoint
Path of structure network checkpoint to resume training from. Use None to train from scratch.
#### learningrate
The optimizer learning rate.
#### learningrate_decay_factor
Decay factor applied to the learning rate.
#### learningrate_decay_step
Number of of optimization steps before learning rate decay is applied.
#### loss_lambda
Reconstruction (L1) loss lambda for structure learning.

---
## Inference parameters
### use_cuda
Activates inference on GPU
### gpu_index
Index of GPU to run networks on.
### segmentation_checkpoint
Path of the segmentation (shape) network checkpoint used for reenactment.
### structure_checkpoint
Path of the structure network checkpoint used for reenactment.
### render_checkpoint
Path of the render (appearance) network checkpoint used for reenactment.
### use_gt_segmentation
If set to true, pseudo gt shape annotations from the dataset will be used for reenactment instead of network estimates
### use_gt_structures
If set to true, pseudo gt structure annotations from the dataset will be used for reenactment instead of network estimates
### num_initial_iterations
Number of initial iterations to compensate first zero input in recurrent components.
### structure_magnification
Constant factor applied to the input structure.
### output_dir
Path of the output directory.
### append_source_image
If set to true, the ground truth source actor image used to extract the input motion sequence will be horizontally concatenated to the final output image for direct comparison.
### generate_segmentations
If true, the estimated shape will be saved to the output directory.
### generate_structures
If true, the estimated structure will be saved to the output directory.
### validation_set
If true, the dataset validation split will be used for reenactment.
### create_videos
If true, a video is automatically generated from the output images using ffmpeg.
### video_framerate
The framerate used for output video generation. 