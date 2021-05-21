'''Config.py: loads a global configuration file for training and testing using the yaml format.'''

import yaml
from munch import DefaultMunch

import utils

__author__ = 'Moritz Kappel'
__email__ = 'kappel@cg.cs.tu-bs.de'

global data


def loadConfig(config_path):
    try:
        yaml_dict = yaml.unsafe_load(open(config_path))
        global data
        data = DefaultMunch.fromDict(yaml_dict)
    except Exception:
        utils.ColorLogger.print('failed to parse config file at location: "{0}"'.format(config_path), 'ERROR')


def saveConfig(config_path):
    try:
        with open(config_path, 'w') as f:
            global data
            yaml.dump(DefaultMunch.toDict(data), f, indent=4, canonical=False)
    except Exception:
        utils.ColorLogger.print('failed to save file at location: "{0}"'.format(config_path), 'ERROR')


def printConfigToTensorboard(config_path, tensorboard_writer):
    with open(config_path, 'r') as file:
        tensorboard_writer.add_text("config", file.read(), global_step=None, walltime=None)
