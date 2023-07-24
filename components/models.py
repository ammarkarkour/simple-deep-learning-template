from torchvision.models import resnet18, ResNet18_Weights


class Models:
    """
    Register the used models here, so they can be accessed later in Config.
    """
    Resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)