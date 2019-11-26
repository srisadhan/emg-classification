import collections
import deepdish as dd
import numpy as np
from pathlib import Path
import yaml
import time
import pandas as pd
import sys
import matplotlib.pyplot as plt

def convert_from_posix_to_std(time_vector):
    """ Convert from posix time format to float values

    Parameters
    ----------
    time_vector : input time is in posix format but and in microseconds

    Returns
    -------
    Time : a vector of float values represented by 60 * 60 * HH + 60 * MM + SS + MillS / 1e6

    """
    Time = np.zeros((1,len(time_vector)))
    i    = 0
    for value in time_vector:
        temp = time.strftime('%H:%M:%S', time.localtime(value / 1e6)) # microseconds to seconds conversion
        temp = temp.split(':')
        Time[0,i] = float(temp[0]) * 60 * 60 + float(temp[1]) * 60 + float(temp[2]) + int(value % 1e6) / 1e6
        i += 1
    return Time

def read_Pos_Force_data(subject):
    """ Read position and force data from the files and save it into a dictionary

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
    file_path = Path(__file__).parents[2] / 'data' / 'raw' / 'force_data' / subject

    data        = collections.defaultdict(dict)

    for file in file_path.iterdir():
        trial = file.name.split('_')

        # Get the time data
        time_data = np.genfromtxt(file,
                                 dtype=None,
                                 delimiter=',',
                                 unpack=True,
                                 usecols=0,
                                 skip_header=2,
                                 encoding=None)

        time_vec = np.zeros((len(time_data),1))
        i = 0

        # Convert the std time vector to float
        for value in time_data:
            temp = value.split(':')
            time_vec[i,0]  = float(temp[0]) * 60 * 60 + float(temp[1]) * 60 + float(temp[2]) + float(temp[3]) / 1e6
            i += 1

        # Read the remaining values
        Force   = np.genfromtxt(file, dtype=float, delimiter=',', unpack=True, usecols=[13,14], skip_header=2, encoding=None).tolist()
        Pos     = np.genfromtxt(file, dtype=float, delimiter=',', unpack=True, usecols=[19,20], skip_header=2, encoding=None).tolist()

        data['Force'][trial[1]]     = Force
        data['Pos'][trial[1]]       = Pos
        data['Time'][trial[1]]      = time_vec
        data['Window'][trial[1]]    = [time_vec[200,0], time_vec[-200,0]]

    return data

def read_EMG_data(subject, time_window, trials):
    """ Read the EMG data and save it into a dictionary
        EMG data is recorded collectively for all the four experiments. It has to be split using the timestamps of the trials
    Parameters
    ----------
    subject : str
        A string of subject ID e.g. 7707.
    time_window : array
        An array of two values [Start_time, End_time]
    trials : str
        Trial information passed as a string e.g. 'HighFine', 'LowGross'.

    Returns
    -------
    data : a dictionary containing EMG and time data for each trial

    """

    file_path = Path(__file__).parents[2] / 'data' / 'raw' / 'emg_data' / subject

    total_data = collections.defaultdict(dict)
    data = collections.defaultdict(dict)

    for file in file_path.iterdir():
        temp = file.name.split('-')

        if temp[0] == 'EMG':
            # Get the EMG and time data
            total_data = np.genfromtxt(file,
                                     dtype=float,
                                     delimiter=',',
                                     unpack=True,
                                     usecols=np.arange(0,9),
                                     skip_header=2)

            # maximum muscle activation for the experiment
            # max_act = np.amax(np.absolute(total_data[1:9, :]), axis=1)
            # Normalize the emg data using the maximum muscle activation
            # total_data[1:9, :] = np.true_divide(total_data[1:9, :], max_act.reshape((8,1)))

            # convert the time from posix
            total_data[0] = convert_from_posix_to_std(total_data[0])
            time_data = np.zeros((1,len(total_data[0,:])))
            time_data = total_data[0,:]
            for trial in trials:
                indices = np.where((time_window[trial][0] <= time_data) & (time_data <= time_window[trial][1]))
                data['EMG'][trial]  = np.concatenate(total_data[1:9, indices])
                data['Time'][trial] = np.concatenate(total_data[0, indices])
    return data

def parse_complete_emg_force_data(subjects, trials, config):
    """ Parse the csv files and return a dictionary of data
    Parameters
    ----------
    subject : str
        A string of subject ID e.g. 7707.
    trials  : str
        Trial information passed as a string e.g. 'HighFine', 'LowGross'.
    config  : configuration file 'yml' format

    Returns
    -------
    Saves the data into a dictionary with the following structure
                                                                  HighFine       HighGross       LowFine        LowGross
                    |--------->  PB |-----> Force              |             |              |               |              |
                    |               |-----> Pos                |             |              |               |              |
                    |               |-----> Time               |             |              |               |              |
    data------------|               |-----> Window (start,end) |             |              |               |              |
                    |                                          |             |              |               |              |
                    |               |-----> EMG                |             |              |               |              |
                    |---------> MYO |-----> Time               |             |              |               |              |

    """

    Data = {}
    for subject in subjects:
        # dictionary initialization
        data = collections.defaultdict(dict)

        data['PB']  = read_Pos_Force_data(subject)
        data['MYO'] = read_EMG_data(subject, data['PB']['Window'], trials)

        Data['Subject_'+subject] = data

    return Data

def label_data(trial):
    """label the epochs
    
    Parameters
    ----------
    trial : str
        string representing the task type
    
    Returns
    -------
    category : int
        int representing class label
    """
    # 3-class encoding
    if trial == 'HighFine':
        category = 1
    elif trial == 'LowGross':
        category = 2
    elif (trial == 'HighGross') or (trial == 'LowFine'):
        category = 3

    return category

def pool_emg_data(subjects, trials, config):
    """pools the data from all the subjects
    
    Parameters
    ----------
    subjects : List 
        strings representing subject identifiers
    trials : list
        strings representing the task types
    config : yaml 
        configuration file
    
    Returns
    -------
    emg : numpy array
        a 3d array of epoched raw emg signals (n_trials, n_channels, n_samples)
    labels : numpy array
        a column of task type
    """
    
    data    = parse_complete_emg_force_data(subjects, trials, config)

    emg     = np.empty((0,0,0))
    labels  = np.empty((0,1))

    for subject in config['subjects']:
        X = np.empty((0,0,0))
        y = np.empty((0,1))

        for trial in config['trials']:
            temp1 = np.transpose(data['Subject_'+subject]['MYO']['EMG'][trial])
            temp2 = label_data(trial) * np.ones((temp1.shape[0],1))

            if X.size == 0:
                X = temp1
                y = temp2
            else:
                X = np.append(X, temp1, axis=0)
                y = np.append(y, temp2, axis=0)

        if emg.size == 0:
            emg = X
            labels = y
        else:
            emg = np.append(emg, X, axis=0)
            labels = np.append(labels, y, axis=0)

    return emg, labels

def epoch_raw_emg(subjects, trials, config):
    """Splits the data into epochs using the user defined window length
    
    Parameters
    ----------
    subjects : List 
        strings representing subject identifiers
    trials : list
        strings representing the task types
    config : yaml 
        configuration file
    
    Returns
    -------
    emg : numpy array
        a 3d array of epoched raw emg signals (n_trials, n_channels, n_samples)
    labels : numpy array
        a column of task type
    """
    
    data    = parse_complete_emg_force_data(subjects, trials, config)

    # window increment for epoching the raw emg data
    increment = int(np.round(config['epoch_length'] * (1 - config['overlap']) * config['sfreq']))
    window    = int(config['epoch_length'] * config['sfreq'])
    emg     = np.empty((0,0,0))
    labels  = np.empty((0,1))

    for subject in config['subjects']:
        X = np.empty((0,0,0))
        y = np.empty((0,1))

        for trial in config['trials']:
            temp = data['Subject_'+subject]['MYO']['EMG'][trial]

            for i in range(0, temp.shape[1]-window, increment):
                temp1 = temp[np.newaxis,:,i:i+window]
                temp2 = np.array([[label_data(trial)]])

                if X.size == 0:
                    X = temp1
                    y = temp2
                else:
                    X = np.append(X, temp1, axis=0)
                    y = np.append(y, temp2, axis=0)

        if emg.size == 0:
            emg = X
            labels = y
        else:
            emg = np.append(emg, X, axis=0)
            labels = np.append(labels, y, axis=0)

    return emg, labels
    