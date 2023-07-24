import argparse

from .data import DataConfig
from .model import ModelConfig
from .optimization import OptimConfig
from .logger import LogConfig

parser = argparse.ArgumentParser(description="Project description")

# Data Configuration parameters
parser.add_argument("-tp", "--training-path", required=False, type=str, help="The path to training data")
parser.add_argument("-vp", "--validation-path", required=False, type=str, help="The path to validation data")
parser.add_argument("-bs", "--batch-size", required=False, type=int, help="The size of one training batch size")
parser.add_argument("-ne", "--num-epochs", required=False, type=int, help="The number of training epochs")

# Model Configuration parameters
parser.add_argument("-mn", "--model-name", required=False, type=str, help="The name of the used model")
parser.add_argument("-pw", "--pretrained-weights", required=False, type=str, help="The pretrained weights")

# Optimization Configuration parameters
parser.add_argument("-cn", "--criterion-name", required=False, type=str, help="The name of the used criterion")
parser.add_argument("-on", "--optimizer-name", required=False, type=str, help="The name of the used optimizer")
parser.add_argument("-lr", "--learning-rate", required=False, type=int, help="The used learning rate")

# Logging Configuration parameters
parser.add_argument("-od", "--output-dir", required=False, type=str, help="The path that contains experents logs")
parser.add_argument("-sd", "--save-dir", required=False, type=str, help="The path that we save model weights in")

# Read the commands and save them
commands = parser.parse_args()
if commands.training_path:
    DataConfig.TRAINGIN_PATH = commands.training_path
if commands.validation_path:
    DataConfig.VALIDATION_PATH = commands.validation_path
if commands.batch_size:
    DataConfig.BATCH_SIZE = commands.batch_size
if commands.num_epochs:
    DataConfig.NUM_EPOCHS = commands.epochs

if commands.model_name:
    ModelConfig.MODEL_NAME = commands.model_name
if commands.pretrained_weights:
    ModelConfig.PRETRAINED_WEIGHTS = commands.pretrained_weights

if commands.criterion_name:
    OptimConfig.CRITERION_NAME = commands.criterion_name
if commands.optimizer_name:
    OptimConfig.OPTIMIZER_NAME = commands.optimizer_name
if commands.learning_rate:
    OptimConfig.LEARNING_RATE = commands.learning_rate

if commands.output_dir:
    LogConfig.OUTPUT_DIR = commands.output_dir
if commands.save_dir:
    LogConfig.SAVE_DIR = commands.save_dir