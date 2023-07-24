import sys
sys.path.append('../')

from components.data_processer import Datasets

class DatasetsModules:
    Training_Dataset = Datasets.Training_dataset
    Validation_Dataset = Datasets.Validation_dataset
