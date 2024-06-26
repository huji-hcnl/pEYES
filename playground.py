import os

import numpy as np
import pandas as pd

import src.pEYES as peyes

CWD = os.getcwd()

#########
dataset = peyes.datasets.lund2013(directory=os.path.join(CWD, "output", "datasets"), save=True, verbose=True)
trial1 = dataset[dataset[peyes.constants.TRIAL_ID_STR] == 1]
