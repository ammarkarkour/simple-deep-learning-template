import sys
sys.path.append('../')

from torch.utils import data

from config.data import DataConfig
from modules.data import DataModules


class Dataloaders:
    """
    Register the used dataloaders here, so they can be accessed later in Config.
    """
    Training_dataloader = data.DataLoader(
        DataModules.Training_Dataset, 
        shuffle=True,
        batch_size=DataConfig.BATCH_SIZE, 
        num_workers=DataConfig.NUM_WORKERS, 
        pin_memory=True,
    )
    
    Validation_dataloader = data.DataLoader(
        DataModules.Validation_Dataset, 
        shuffle=False,
        batch_size=DataConfig.MINI_BATCH_SIZE, 
        num_workers=DataConfig.NUM_WORKERS, 
        pin_memory=True,
    )