import sys
sys.path.append('../')

from config.data import DataConfig
from components.data_loader import Dataloaders

class DataloadersModules:
    Num_epochs = 50
    Training_Dataloader = Dataloaders.Training_dataloader
    Validation_Dataloader = Dataloaders.Validation_dataloader
    Update_step = DataConfig.BATCH_SIZE // DataConfig.MINI_BATCH_SIZE