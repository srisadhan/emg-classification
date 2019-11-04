import collections
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import deepdish as dd
import sys

def get_trial_path(subject, trial, config, robot=False):
    """Get the trail path for a given subject and trial.

    Parameters
    ----------
    subject : str
        A string of subject ID e.g. 7707.
    trial : str
        A trail e.g. HighFine..
    config : yaml
        The configuration file.
    robot : bool
        To get robot path or emg path

    Returns
    -------
    str
        A str specifying the path to the data.

    """
    if robot:
        # Trial time
        path = Path(__file__).parents[2] / config['force_data_path'] / subject
        for file in path.iterdir():
            file_name = file.name.split('_')
            if file_name[1] == trial:
                break
        trial_path = file

    else:

        # Trial time
        path = Path(__file__).parents[2] / config['raw_emg_path'] / subject
        for file in path.iterdir():
            file_name = file.name.split('-')
            if file_name[0] == 'emg':
                break
        trial_path = file

    return trial_path


def get_trail_time(subject, trial, config):
    """Get the trail time for a given subject and trial.

    Parameters
    ----------
    subject : str
        A string of subject ID e.g. 7707.
    trial : str
        A trail e.g. HighFine..
    config : yaml
        The configuration file.

    Returns
    -------
    str
        A str specifying the path to the data.

    """

    # Trial time
    path = Path(__file__).parents[2] / config['force_data_path'] / subject
    for file in path.iterdir():
        file_name = file.name.split('_')
        if file_name[1] == trial:
            break
    trial_path = file

    # Get the trial time
    trial_time = np.genfromtxt(trial_path,
                               dtype=str,
                               delimiter=',',
                               usecols=0,
                               skip_footer=150,
                               skip_header=100).tolist()

    # Get EMG time
    trial_path = get_trial_path(subject, trial, config)
    time_data = np.genfromtxt(trial_path,
                              dtype=str,
                              delimiter=',',
                              usecols=0,
                              skip_footer=150,
                              skip_header=100).tolist()

    # Get the sampling frequency
    emg_time = [
        datetime.fromtimestamp(float(item) / 1e6) for item in time_data
    ]
    dt = np.diff(emg_time).mean()  # average sampling rate
    sfreq = 1 / dt.total_seconds()

    # Update year, month, and day of start time
    start_t = datetime.strptime(trial_time[0], '%H:%M:%S:%f')
    start_t = start_t.replace(year=emg_time[0].year,
                              month=emg_time[0].month,
                              day=emg_time[0].day)

    # Update year, month, and day of end time
    end_t = datetime.strptime(trial_time[-1], '%H:%M:%S:%f')
    end_t = end_t.replace(year=emg_time[0].year,
                          month=emg_time[0].month,
                          day=emg_time[0].day)

    trial_start = (start_t - emg_time[0]).total_seconds()  # convert to seconds
    trial_end = (end_t - emg_time[0]).total_seconds()

    return trial_start, trial_end, sfreq


def get_raw_emg(subject, trial, config):
    """Get the raw emg data for a subject and trail.

    Parameters
    ----------
    subject : str
        A string of subject ID e.g. 7707.
    trial : str
        A trail e.g. HighFine.
    config : yaml
        The configuration file.

    Returns
    -------
    mne object
        A raw mne object.

    """
    # Get trail path
    trial_path = get_trial_path(subject, trial, config)

    # Get the EMG data
    emg_data = np.genfromtxt(trial_path,
                             dtype=float,
                             delimiter=',',
                             unpack=True,
                             usecols=[1, 2, 3, 4, 5, 6, 7, 8],
                             skip_footer=150,
                             skip_header=100)

    trial_start, trial_end, sfreq = get_trail_time(subject, trial, config)

    # Create mne raw object
    info = mne.create_info(ch_names=[
        'emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6', 'emg_7', 'emg_8'
    ],
                           ch_types=['misc'] * emg_data.shape[0],
                           sfreq=sfreq)

    # Create mne raw file
    raw = mne.io.RawArray(emg_data, info, verbose=False)

    # Additional information
    raw.info['subject_info'] = subject
    raw.info['experimenter'] = 'hemanth'

    return raw, [trial_start, trial_end]


def get_robot_data(subject, trial, config):
    """Get the force and velocity data of a subject and a trial.

    Parameters
    ----------
    subject : str
        A string of subject ID e.g. 7707.
    trial : str
        A trail e.g. HighFine..
    config : yaml
        The configuration file.
    Returns
    ----------
    robot_data : array
        A numpy array containing  total_force, velocity

    """
    trial_path = get_trial_path(subject, trial, config, robot=True)
    data = np.genfromtxt(trial_path,
                         dtype=float,
                         delimiter=',',
                         usecols=[13, 14, 19, 20],
                         skip_footer=1250,
                         skip_header=1250)
    time_data = np.genfromtxt(trial_path,
                              dtype=str,
                              delimiter=',',
                              usecols=0,
                              skip_footer=1250,
                              skip_header=1250).tolist()
    # Total force
    total_force = np.linalg.norm(data[1:, 0:2], axis=1)

    # Average time
    time = [datetime.strptime(item, '%H:%M:%S:%f') for item in time_data]
    time = np.array(time)  # convert to numpy

    # Convert to seconds
    helper = np.vectorize(lambda x: x.total_seconds())
    dt = helper(np.diff(time))  # average time difference

    # x and y co-ordinates of the end effector
    dx = np.diff(data[:, 2])
    dy = np.diff(data[:, 3])
    velocity = np.sqrt(np.square(dx) + np.square(dy)) / dt

    # Stack all the vectors
    robot_data = np.vstack((total_force, velocity)).T

    return robot_data


def create_emg_data(subjects, trials, config):
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
    emg_data = {}
    # Loop over all subjects and error types
    for subject in subjects:
        data = collections.defaultdict(dict)
        for trial in trials:
            raw_data, trial_time = get_raw_emg(subject, trial, config)
            data['emg'][trial] = raw_data
            data['time'][trial] = trial_time
        emg_data['subject_' + subject] = data

    return emg_data


def get_emg_epoch(raw_emg, time, config):
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
    # Parameters
    epoch_length = config['epoch_length']
    overlap = config['overlap']

    raw_cropped = raw_emg.copy().crop(tmin=time[0], tmax=time[1])
    events = mne.make_fixed_length_events(raw_cropped,
                                          duration=epoch_length,
                                          overlap=epoch_length * overlap)
    epochs = mne.Epochs(raw_cropped,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        verbose=False)
    return epochs


def create_emg_epoch(subjects, trials, config):
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
            temp['emg'][trial] = get_emg_epoch(raw_emg, time, config)
        emg_epochs['subject_' + subject] = temp

    return emg_epochs


def create_robot_dataframe(config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels, tags
        2 arrays features and labels.
        A tag determines whether the data point is used in training.

    """

    columns = ['total_force', 'velocity', 'task', 'damping', 'subject']
    dataframe = pd.DataFrame(columns=columns)

    for subject in config['subjects']:
        for trial in config['trials']:
            temp = np.mean(get_robot_data(subject, trial, config), axis=0)

            # Form a dataframe and sample the array simultaneously
            temp_df = pd.DataFrame(data=np.expand_dims(temp, axis=0),
                                   columns=['total_force', 'velocity'])

            # Add other information
            temp_df['subject'] = subject
            if trial.find('Gross') > 0:
                temp_df['task'] = 'Gross'
            else:
                temp_df['task'] = 'Fine'

            if trial.find('High'):
                temp_df['damping'] = 'Low'
            else:
                temp_df['damping'] = 'High'

            # Append to the main dataframe
            dataframe = pd.concat([dataframe, temp_df],
                                  ignore_index=True,
                                  sort=False)

    return dataframe


def sort_order_emg_channels(config):
    """ Sort the emg channels of each subject based on the decreasing variance
    Parameters
    ----------
    config : yaml
        configuration file
    sort   : bool
        sort the channels if True

    Return
    ------

    """
    Data = collections.defaultdict(dict)

    for subject in config['subjects']:
        path = Path(__file__).parents[2] / config['raw_emg_path'] / subject
        for file in path.iterdir():
            file_name = file.name.split('-')
            if file_name[0] == 'emg':
                data    = np.genfromtxt(file,
                                     dtype=float,
                                     delimiter=',',
                                     usecols=np.arange(1,9,1),
                                     skip_footer=1250,
                                     skip_header=1250)
                emg_std  = np.std(data, axis=0)
                emg_order= np.argsort(-emg_std) # decreasing order

        Data['subject_'+ subject]['channel_order'] = emg_order

    return Data
