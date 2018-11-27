from .policy import Policy
import numpy as np


class PushingPolicy(Policy):
    def __init__(self, hparam_overrides):
        self._hp = self._default_hparams()

    def act(self, state):
        act = np.zeros(3)
        return {'actions': act}