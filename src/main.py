import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import yaml

from data.clean_data import clean_epoch_data
from data.create_data import create_emg_data, create_emg_epoch
from utils import *

# The configuration file
config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)

with skip_run('skip', 'create_emg_data') as check, check():
    data = create_emg_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['raw_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'create_epoch_data') as check, check():
    data = clean_emg_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['epoch_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('run', 'clean_epoch_data') as check, check():
    data = clean_emg_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['clean_emg_data']
    save_data(str(save_path), data, save=True)
