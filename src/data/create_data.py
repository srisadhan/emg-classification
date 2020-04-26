import collections
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import deepdish as dd
import sys
import math
from mne.datasets import sample
from sklearn.decomposition import PCA

def get_trial_path(subject, trial, config, sensor):
    """Get the trail path for a given subject and trial.

    Parameters
    ----------
    subject : str
        A string of subject ID e.g. 7707.
    trial : str
        A trail e.g. HighFine..
    config : yaml
        The configuration file.
    sensor : string
        string representing the type of file to be retrieved. For eg. 'EMG', 'PB', 'IMU'
        
    Returns
    -------
    str
        A str specifying the path to the data.

    """
    if sensor == 'PB':
        # Trial time
        path = Path(__file__).parents[2] / config['force_data_path'] / subject
        for file in path.iterdir():
            file_name = file.name.split('_')
            if file_name[1] == trial:
                break
        trial_path = file

    elif sensor == 'EMG':
        # Trial time
        path = Path(__file__).parents[2] / config['raw_emg_path'] / subject
        for file in path.iterdir():
            file_name = file.name.split('-')
            if file_name[0].upper() == 'EMG':
                break
        trial_path = file

    elif sensor == 'IMU':
        # Trial time
        path = Path(__file__).parents[2] / config['raw_emg_path'] / subject
        for file in path.iterdir():
            file_name = file.name.split('-')
            if file_name[0].lower() == 'accelerometer':
                break
        trial_path = file
        
    return trial_path


def get_trial_time(subject, trial, config):
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
                               skip_footer=config['skip_footer'],
                                skip_header=config['skip_header']).tolist()

    # Get EMG time
    trial_path = get_trial_path(subject, trial, config, 'EMG')
    time_data = np.genfromtxt(trial_path,
                            dtype=str,
                            delimiter=',',
                            usecols=0,
                            skip_footer=config['skip_footer'],
                            skip_header=config['skip_header']).tolist()

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
    trial_path = get_trial_path(subject, trial, config, 'EMG')

    # Get the EMG data
    emg_data = np.genfromtxt(trial_path,
                             dtype=float,
                             delimiter=',',
                             unpack=True,
                             usecols=[1, 2, 3, 4, 5, 6, 7, 8],
                             skip_footer=config['skip_footer'],
                             skip_header=config['skip_header'])

    trial_start, trial_end, sfreq = get_trial_time(subject, trial, config)

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


def convert_time(time):
    """ Convert the time from HH::MM::SS::MS to seconds 
    Parameters:
    ----------
    time : numpy array
        time vector (HH::MM::SS::MS)
    
    Returns:
    -------
    time_vec : numpy array
        time vector in seconds
    sfreq : float
        sampling frequency
    """

    time_vec = np.zeros((len(time),1))
    i = 0

    # Convert the std time vector to float
    for value in time:
        temp = value.split(':')
        time_vec[i,0]  = float(temp[0]) * 60 * 60 + float(temp[1]) * 60 + float(temp[2]) + float(temp[3]) / 1e6
        i += 1

    dt = np.diff(time_vec, axis=0).mean()  # average sampling rate
    sfreq = 1 / dt

    return time_vec, sfreq


def get_trial_time_exp2(subject, trial, PB_path, EMG_path, config):
    """Get the trial time using both PB and EMG time windows.

    Parameters
    ----------
    subject : string
        subject identifier. For eg. 7707
    trial : string
        task type (HighFine, LowFine, HighGross, LowGross)
    PB_path : string
        path of the powerball file
    EMG_path : array
        time vector.
    config : yml
        yaml file

    Returns
    -------
    trial_start : float
        start time  of the trial
    trial_end : float
        end time of the trial
    sfreq : float
        sampling frequency
        A str specifying the path to the data.
    """    

    # Get the trial time
    time_PB = np.genfromtxt(PB_path,
                            dtype=str,
                            delimiter=',',
                            usecols=0,
                            unpack=True,
                            skip_footer=config['skip_footer'],
                            skip_header=config['skip_header'])
    
    time_EMG = np.genfromtxt(EMG_path,
                            dtype=str,
                            delimiter=',',
                            usecols=0,
                            unpack=True,
                            skip_footer=config['skip_footer'],
                            skip_header=config['skip_header'])

    time_EMG, sfreq = convert_time(time_EMG)
    time_PB, _ = convert_time(time_PB)

    trial_start = time_EMG[0]
    trial_end   = time_EMG[-1]
    
    if trial_start < time_PB[0]:
        trial_start = time_PB[0]
    elif trial_end > time_PB[-1]:
        trial_end = time_PB[-1]

    return trial_start, trial_end, sfreq


def get_raw_emg_exp2(subject, trial, config):
    """Get the raw emg data for a subject and trail from experiment 2

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
    # path of the PB file
    # if subject in config['subjects2']:
    #     trial_path = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial / 'PB.csv'
    # else:
    #     trial_path = get_trial_path(subject, trial, config, 'PB')

    # path of the files
    # if subject in config['subjects2']:
    PB_path  = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial / 'PB.csv'
    EMG_path = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial / 'EMG.csv'
    # else:
    #     PB_path  = get_trial_path(subject, trial, config, 'PB')
    #     EMG_path = get_trial_path(subject, trial, config, 'EMG')

    # path of the EMG file
    filepath = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial

    if filepath.exists():
        for file in filepath.iterdir():
            if (file.name.split('.')[0] == 'EMG'):
                # Get the time from EMG.csv
                time = np.genfromtxt(file,
                                    dtype=None,
                                    delimiter=',',
                                    unpack=True,
                                    skip_footer=config['skip_footer'],
                                    skip_header=config['skip_header'],
                                    usecols=0,
                                    encoding=None)
                # Get the EMG data
                EMG_data = np.genfromtxt(file,
                                    dtype=float,
                                    delimiter=',',
                                    unpack=True,
                                    skip_footer=config['skip_footer'],
                                    skip_header=config['skip_header'],
                                    usecols=np.arange(1,9),
                                    encoding=None)
                
                # get the actual trial start and end time based on the PB and MYO data
                if subject in config['subjects2']:
                    trial_start, trial_end, _ = get_trial_time_exp2(subject, trial, PB_path, EMG_path, config)
                else:
                    trial_start, trial_end, _ = get_trial_time(subject, trial, config)

                time_EMG, sfreq = convert_time(time)

                indices = np.all([time_EMG >= trial_start, time_EMG <= trial_end], axis=0)
                
                time_EMG = time_EMG[indices[:,0]]
                EMG_data = EMG_data[:, indices[:,0]]

                # Create mne raw object
                info = mne.create_info(ch_names=[
                    'emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6', 'emg_7', 'emg_8'
                ],
                                    ch_types=['misc'] * EMG_data.shape[0],
                                    sfreq=sfreq)

                # Create mne raw file
                raw = mne.io.RawArray(EMG_data, info, verbose=False)

                # Additional information
                raw.info['subject_info'] = subject

                return raw, [trial_start[:], trial_end[:]]
    else:
        return [], []


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
            if subject in config['subjects2']:
                raw_data, trial_time = get_raw_emg_exp2(subject, trial, config)
            else:
                raw_data, trial_time = get_raw_emg(subject, trial, config)
                
            if raw_data :
                data['EMG'][trial] = raw_data
                data['time'][trial] = trial_time

        emg_data['subject_' + subject] = data
    
    return emg_data


def get_emg_epoch(subject,raw_emg, time, config):
    """Create the epoch data from raw data.

    Parameter
    ----------
    subject : string
        String of subject ID e.g. 7707
    raw_emg : mne raw object
        data structure of raw_emg
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
    
    if subject in config['subjects2']:
        raw_cropped = raw_emg.copy().resample(config['sfreq2'], npad='auto', verbose='error')
    else:
        raw_cropped = raw_emg.copy().crop(tmin=time[0], tmax=time[1])
        raw_cropped = raw_cropped.copy().resample(config['sfreq2'], npad='auto', verbose='error')

    if config['high_pass_filter']:
        raw_cropped = raw_cropped.filter(l_freq=config['filter_freq'][0], h_freq=None, picks=['emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6', 'emg_7', 'emg_8'])

    events = mne.make_fixed_length_events(raw_cropped,
                                          duration=epoch_length,
                                          overlap=epoch_length * overlap)
    epochs = mne.Epochs(raw_cropped,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=None,
                        verbose=False)
    return epochs


def create_emg_epoch(subjects, trials, read_path, config):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : list
        String of subject ID e.g. 7707
    error_type : list
        Types of trials i.e., e.g. HighFine.
    read_path : string
        path of the saved file
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
    # read_path = Path(__file__).parents[2] / config['raw_emg_data']
    data = dd.io.load(str(read_path))

    # Loop over all subjects and error types
    for subject in subjects:
        temp = collections.defaultdict(dict)
        if subject in config['test_subjects']:
            for trial in config['trials']:
                raw_emg = data['subject_' + subject]['EMG'][trial]
                time = data['subject_' + subject]['time'][trial]
                
                # Create epoch data
                temp['EMG'][trial] = get_emg_epoch(subject,raw_emg, time, config)
                temp['time'][trial] = time
        else:
            for trial in ['HighFine', 'LowGross', 'HighGross', 'LowFine']:
                raw_emg = data['subject_' + subject]['EMG'][trial]
                time = data['subject_' + subject]['time'][trial]
                # Create epoch data
                temp['EMG'][trial] = get_emg_epoch(subject,raw_emg, time, config)
                temp['time'][trial] = time

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
            if file_name[0] == 'EMG':
                data    = np.genfromtxt(file,
                                     dtype=float,
                                     delimiter=',',
                                     usecols=np.arange(1,9,1),
                                     skip_footer=config['skip_footer'],
                                     skip_header=config['skip_header'])
                emg_std  = np.std(data, axis=0)
                emg_order= np.argsort(-emg_std) # decreasing order

        Data['subject_'+ subject]['channel_order'] = emg_order

    return Data


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
    trial_path = get_trial_path(subject, trial, config, 'PB')
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


def get_PB_data(subject, trial, config):
    """Get the force and position data of a subject and a trial.

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
        A numpy array containing  total_force

    """
    # path of the files
    if subject in config['subjects2']:
        PB_path  = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial / 'PB.csv'
        EMG_path = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial / 'EMG.csv'
    else:
        if trial not in config['comb_trials']:
            PB_path  = get_trial_path(subject, trial, config, 'PB')
            EMG_path = get_trial_path(subject, trial, config, 'EMG')
    
    # read the data
    PB_data = np.genfromtxt(PB_path,
                            dtype=float,
                            delimiter=',',
                            unpack=True,
                            usecols=[16, 17, 19, 20], # [13, 14]-Fx,Fy; [16,17]-Mx,My
                            skip_footer=config['skip_footer'],
                            skip_header=config['skip_header'])
    time_data = np.genfromtxt(PB_path,
                            dtype=str,
                            delimiter=',',
                            unpack=True,
                            usecols=0,
                            skip_footer=config['skip_footer'],
                            skip_header=config['skip_header'])

    # Moments are used in Amir's experiment. The moment to force conversion scalar provided in the experiment are [-10.0,-10.0]
    PB_data[0:2,:] = np.multiply(PB_data[0:2,:], np.array([-10, 10]).reshape(2,1)) # only when 16, 17 are used instead of 13, 14

    # Calculate the forces in the tangential and the normal directions
    PB_data = tangential_normal_force_components(PB_data)

    PB_data = PCA_force_components(PB_data, wind_len=4)
    
    # get the actual trial start and end time based on the PB and MYO data
    if subject in config['subjects2']:
        trial_start, trial_end, _ = get_trial_time_exp2(subject, trial, PB_path, EMG_path, config)
        time_data, sfreq = convert_time(time_data)
        
        indices = np.all([time_data >= trial_start, time_data <= trial_end], axis=0)

        time_data = time_data[indices[:,0]]
        PB_data   = PB_data[:,indices[:,0]]

    else:
        trial_start, trial_end, _ = get_trial_time(subject, trial, config)
        time_data, sfreq = convert_time(time_data)
    
    
    # creating an mne object
    info = mne.create_info(ch_names=['Fx', 'Fy', 'X', 'Y', 'Ft', 'Fn', 'F_pca1', 'F_pca2'], sfreq=sfreq, ch_types=['misc'] * 8)
    raw = mne.io.RawArray(PB_data, info, verbose=False)

    return raw, time_data


def create_PB_data(subjects, trials, config):
    """Create the force data with each subject data in a dictionary.

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
        data['PB'] = Fx, Fy, X, Y, Ft, Fn
    """
    PB_data = {}
    # Loop over all subjects and error types
    for subject in subjects:
        data = collections.defaultdict(dict)

        if subject not in config['test_subjects']:
            for trial in (set(config['trials']) - set(config['comb_trials'])):
                raw_data, time = get_PB_data(subject, trial, config)   
                data['PB'][trial] = raw_data
                data['time'][trial] = time
        else:
            # for trial in config['comb_trials']:
            for trial in config['trials']:
                raw_data, time = get_PB_data(subject, trial, config)   
                data['PB'][trial] = raw_data
                data['time'][trial] = time


        PB_data['subject_' + subject] = data
        
    return PB_data


def get_PB_epoch(subject,raw_data, config):
    """Create the epoch data from raw data.

    Parameter
    ----------
    subject : string
        String of subject ID e.g. 7707
    raw_emg : mne raw object
        data structure of raw_emg
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
    
    raw_cropped = raw_data.copy().resample(config['sfreq2'], npad='auto', verbose='error')

    events = mne.make_fixed_length_events(raw_cropped,
                                          duration=epoch_length,
                                          overlap=epoch_length * overlap)
    epochs = mne.Epochs(raw_cropped,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=None,
                        verbose=False)
    return epochs


def create_PB_epoch(subjects, trials, config):
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
    data_epochs = {}

    # Load the data
    read_path = Path(__file__).parents[2] / config['raw_PB_data']
    data = dd.io.load(str(read_path))

    # Loop over all subjects and error types
    for subject in subjects:
        temp = collections.defaultdict(dict)

        if subject not in config['test_subjects']:
            for trial in (set(config['trials']) - set(config['comb_trials'])):
                raw_data = data['subject_' + subject]['PB'][trial]
                time = data['subject_' + subject]['time'][trial]

                # Create epoch data
                temp['PB'][trial] = get_PB_epoch(subject, raw_data, config)
                temp['time'][trial] = time
        else:
            # for trial in config['comb_trials']:
            for trial in config['trials']:
                raw_data = data['subject_' + subject]['PB'][trial]
                time = data['subject_' + subject]['time'][trial]

                # Create epoch data
                temp['PB'][trial] = get_PB_epoch(subject, raw_data, config)
                temp['time'][trial] = time

        data_epochs['subject_' + subject] = temp

    return data_epochs

def PCA_force_components(data, wind_len=10):
    """Perform the PCA on the history of points to capture the 
    dominant directions of the force and project the force data 
    in that direction

    Parameters
    ----------
    data : nd-array
        force and position data from the robot
    wind_len : int, optional
        wind_len to consider the history of points, by default 4
    
    Return
    ------
    data : nd-array
        data concatenated to the array
    """        
    force_xy    = data[0:2,:]

    force_PCA   = np.zeros(force_xy.shape)
    pca = PCA(copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
    
    for i in range(force_xy.shape[0]):
        if i < wind_len:
            force_PCA[i, :] = force_xy[i, :]
        else:
            force_PCA[i, :] = pca.fit(force_xy[i-wind_len:i+1,:]).transform(force_xy[i,:])

    return(np.concatenate((data[:,:], force_PCA[:,:]), axis=0))
    
def tangential_normal_force_components(data):
    """Calculates the tangential and the normal components of the force

    Arguments:
        data {ndarray} -- a 4-dimensional array consisting of Fx, Fy, X, Y as rows

    Returns:
        data {ndarray} -- a 6-dimensional array consisting of Fx, Fy, X, Y, Ft, Fn as rows
    """
    force_xy    = data[0:2,:]
    pos_xy      = data[2:4,:]

    for i in range(5, pos_xy.shape[0]):
        #FIXME: I should not be using this because I don't have future information
        # pos_xy[i,0] = np.ma.average(pos_xy[i-2:i+3, 0])
        # pos_xy[i,1] = np.ma.average(pos_xy[i-2:i+3, 1])

        pos_xy[i,0] = np.ma.average(pos_xy[i-5:i, 0])
        pos_xy[i,1] = np.ma.average(pos_xy[i-5:i, 1])
        
    # i dont think fitting a polynomial and finding tangent is appropriate
    # for i in range(2, pos.shape[0]-2):
    #     p       = np.polyfit(pos[i-2:2:i+3,0], pos[i-2:2:i+3,1], 3)
    #     ptan    = 3 * p[0] * pos[i,0]**2 + 2 * p[1] * pos[i,0] + p[2]
        
    #     tang_angle  = math.atan2(ptan, pos[i,0])
    #     force_angle = math.atan2(force_xy[i,1], force_xy[i,0])

    pos_diff    = np.diff(pos_xy, axis=1)
    force_t     = np.zeros((1,force_xy.shape[1]))
    force_n     = np.zeros((1,force_xy.shape[1]))

    for i in range(1,force_xy.shape[1]):
        mag = np.linalg.norm(force_xy[:,i])
        ang = angle_between(pos_diff[:,i-1], force_xy[:,i])
        
        force_t[0,i]    = mag * math.cos(ang) 
        force_n[0,i]    = mag * math.sin(ang) 
    
    return(np.concatenate((data[:,:], force_t, force_n), axis=0))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    vec_norm = np.linalg.norm(vector)

    if vec_norm == 0:
        return vector * 0
    else:
        return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    
    Arguments:
        v1 {numpy array} -- a 1d array
        v2 {numpy array} -- a 1d array
    
    Returns:
        [type] -- [description]
    """ 
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_IMU_data(subject, trial, config):
    """Get the force and position data of a subject and a trial.

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
        A numpy array containing  total_force

    """
    # path of the files
    if subject in config['subjects2']:
        IMU_path  = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial / 'IMU.csv'
        EMG_path = Path(__file__).parents[2] / config['exp2_data_path'] / subject / trial / 'EMG.csv'
        cols = [5,6,7]
    else:
        if trial not in config['comb_trials']:
            IMU_path  = get_trial_path(subject, trial, config, 'IMU')
            EMG_path = get_trial_path(subject, trial, config, 'EMG')    
            cols = [1,2,3]
    
    # read the data
    IMU_data = np.genfromtxt(IMU_path,
                            dtype=float,
                            delimiter=',',
                            unpack=True,
                            usecols=cols, # Ax, Ay, Az
                            skip_footer=config['skip_footer'],
                            skip_header=config['skip_header'])
    time_data = np.genfromtxt(IMU_path,
                            dtype=str,
                            delimiter=',',
                            unpack=True,
                            usecols=0,
                            skip_footer=config['skip_footer'],
                            skip_header=config['skip_header'])

    # get the actual trial start and end time based on the PB and MYO data
    if subject in config['subjects2']:
        trial_start, trial_end, _ = get_trial_time_exp2(subject, trial, IMU_path, EMG_path, config)
        time_data, sfreq = convert_time(time_data)
        indices = np.all([time_data >= trial_start, time_data <= trial_end], axis=0)

        time_data = time_data[indices[:,0]]
        # only normalize the IMU data for subjects in the subjects2 list, Amir's data is already normalized
        IMU_data  = IMU_data[:,indices[:,0]] / config['IMUAccScale'] 

    else:
        trial_start, trial_end, _ = get_trial_time(subject, trial, config)
        sfreq = 50 # 50 Hz for the IMU data
    # creating an mne object
    info = mne.create_info(ch_names=['Ax', 'Ay', 'Az'], sfreq=sfreq, ch_types=['misc'] * 3)
    raw = mne.io.RawArray(IMU_data, info, verbose=False)

    return raw, time_data


def create_IMU_data(subjects, trials, config):
    """Create the IMU data with each subject data in a dictionary.

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
        data['PB'] = Fx, Fy, X, Y, Ft, Fn
    """
    IMU_data = {}
    # Loop over all subjects and error types
    for subject in subjects:
        data = collections.defaultdict(dict)

        if subject not in config['test_subjects']:
            for trial in (set(config['trials']) - set(config['comb_trials'])):
                raw_data, time = get_IMU_data(subject, trial, config)   
                data['IMU'][trial] = raw_data
                data['time'][trial] = time
        else:
            # for trial in config['comb_trials']:
            for trial in config['trials']:
                raw_data, time = get_IMU_data(subject, trial, config)   
                data['IMU'][trial] = raw_data
                data['time'][trial] = time


        IMU_data['subject_' + subject] = data
        
    return IMU_data


def get_IMU_epoch(subject,raw_data, config):
    """Create the epoch data from raw data.

    Parameter
    ----------
    subject : string
        String of subject ID e.g. 7707
    raw_emg : mne raw object
        data structure of raw_emg
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
    
    raw_cropped = raw_data.copy().resample(config['sfreq_IMU'], npad='auto', verbose='error')

    events = mne.make_fixed_length_events(raw_cropped,
                                          duration=epoch_length,
                                          overlap=epoch_length * overlap)
    epochs = mne.Epochs(raw_cropped,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=None,
                        verbose=False)
    return epochs


def create_IMU_epoch(subjects, trials, config):
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
    data_epochs = {}

    # Load the data
    read_path = Path(__file__).parents[2] / config['raw_IMU_data']
    data = dd.io.load(str(read_path))

    # Loop over all subjects and error types
    for subject in subjects:
        temp = collections.defaultdict(dict)

        if subject not in config['test_subjects']:
            for trial in (set(config['trials']) - set(config['comb_trials'])):
                raw_data = data['subject_' + subject]['IMU'][trial]
                time = data['subject_' + subject]['time'][trial]

                # Create epoch data
                temp['IMU'][trial] = get_IMU_epoch(subject, raw_data, config)
                temp['time'][trial] = time
        else:
            # for trial in config['comb_trials']:
            for trial in config['trials']:
                raw_data = data['subject_' + subject]['IMU'][trial]
                time = data['subject_' + subject]['time'][trial]

                # Create epoch data
                temp['IMU'][trial] = get_IMU_epoch(subject, raw_data, config)
                temp['time'][trial] = time

        data_epochs['subject_' + subject] = temp

    return data_epochs

