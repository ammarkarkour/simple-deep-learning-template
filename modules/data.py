import sys
sys.path.append('../')

from config.data import DataConfig
from components.data_processer import Datasets
from components.data_loader import Dataloaders

class DataModules:
    Num_epochs = 50
    Training_Dataset = Datasets.Training_dataset
    Validation_Dataset = Datasets.Validation_dataset
    Training_Dataloader = Dataloaders.Training_dataloader
    Validation_Dataloader = Dataloaders.Validation_dataloader
    Update_step = DataConfig.BATCH_SIZE // DataConfig.MINI_BATCH_SIZE