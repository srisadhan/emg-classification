import yaml
import collections
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import deepdish as dd


def create_emg_epoch(raw_emg, time, config):
    """Create the epoch data from raw data.

    Parameter
    ----------
    raw_emg : mne raw object
        String of subject ID e.g. 7707
    time : list
        A list with start and end time
    config : yaml
        The configuration file

    Returns
    ----------
    mne epoch data
        A data (dict) of all the subjects with different conditions

    """

    raw_cropped = raw_emg.copy().crop(tmin=time[0], tmax=time[1])
    events = mne.make_fixed_length_events(raw_cropped,
                                          duration=config['epoch_length'])
    epochs = mne.Epochs(raw_cropped,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        verbose=False)
    return epochs


def clean_emg_data(subjects, trials, config):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : list
        String of subject ID e.g. 7707
    error_type : list
        Types of trials i.e., e.g. HighFine.
    config : yaml
        The configuration file

    Returns
    ----------
    dict
        A data (dict) of all the subjects with different conditions

    """

    # Empty dictionary
    emg_epochs = {}

    # Load the data
    read_path = Path(__file__).parents[2] / config['raw_emg_data']
    data = dd.io.load(str(read_path))

    # Loop over all subjects and error types
    for subject in subjects:
        temp = collections.defaultdict(dict)
        for trial in trials:
            raw_emg = data['subject_' + subject]['emg'][trial]
            time = data['subject_' + subject]['time'][trial]

            # Create epoch data
            temp['emg'][trial] = create_emg_epoch(raw_emg, time, config)
        emg_epochs['subject_' + subject] = temp

    return emg_epochs
