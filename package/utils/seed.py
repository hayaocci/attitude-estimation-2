import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from pathlib import Path
import yaml
import argparse
from math import pi

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)