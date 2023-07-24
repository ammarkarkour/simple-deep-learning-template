import sys
sys.path.append('../')

import torch

from modules.model import ModelModules
from config.optimization import OptimConfig


class Optimizers:
    """
    Register the used optimizers here, so they can be accessed later in Config.
    """
    SGD = torch.optim.SGD(
        ModelModules.Model.parameters(),
        lr=OptimConfig.LEARNING_RATE,
        momentum=0.9
    )