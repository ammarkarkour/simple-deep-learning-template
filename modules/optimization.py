import sys
sys.path.append('../')

from components.optimizers import Optimizers
from components.criterions import Criterions

class OptimModules:
    Criterion = Criterions.CrossEntropy
    Optimizer = Optimizers.SGD