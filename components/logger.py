import sys
sys.path.append('../')

from torch.utils.tensorboard import SummaryWriter

from config.logger import LogConfig


class Loggers:
    """
    Register the used loggers here, so they can be accessed later in Config.
    """
    TensorboardWriter = SummaryWriter(LogConfig.OUTPUT_DIR)
