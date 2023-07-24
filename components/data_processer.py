import sys
sys.path.append('../')

import torchvision
from torchvision.transforms import *

from config.data import DataConfig


class Transformations:
    """
    A class that contains transforms stack for image Augmentations.
    """
    Training_transforms = transforms.Compose([
        transforms.RandomAffine(30, (0.3, 0.3), (0.8, 1.2)),
        transforms.ToTensor(),
    ])

    Validations_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

class Datasets:
    """
    Register the used datasets here, so they can be accessed later in Config.
    """
    Training_dataset = torchvision.datasets.ImageFolder(
        root=DataConfig.TRAINING_PATH,
        transform=Transformations.Training_transforms,
    )

    Validation_dataset = torchvision.datasets.ImageFolder(
        root=DataConfig.VALIDATION_PATH,
        transform=Transformations.Validation_transforms,
    )