import torch


class Criterions:
    """
    Register the used criterions here, so they can be accessed later in Config.
    """
    CrossEntropy = torch.nn.CrossEntropyLoss()