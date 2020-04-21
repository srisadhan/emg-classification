import collections
from pathlib import Path

import deepdish as dd
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

def one_hot_encode(label_length, category):
    """Generate one hot encoded value of required length and category.

    Parameters
    ----------
    label_length : int
        required lenght of the array.
    category : int
        Caterory e.g: category=2, [0, 1, 0] in 3 class system

    Returns
    -------
    array
        One hot encoded array.

    """
    y = np.zeros((label_length, len(category)))
    y[:, category.index(1)] = 1

    return y


def convert_to_array(subject, trial, path, sensor, config):
    """Converts the edf files in eeg and robot dataset into arrays.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trial : str
        Trail e.g. HighFine, LowGross.
    config : yaml
        The configuration file.
    n_class: int
        The number of classes
    path : string
        The path of file
    sensor : string
        The selection of data type between 'EMG' and 'PB'

    Returns
    -------
    array
        An array of feature (x) and lables (y)

    """

    # Read path
    # emg_path = str(Path(__file__).parents[2] / config['epoch_emg_data'])

    # Load the data
    data = dd.io.load(path, group='/' + 'subject_' + subject)
    epochs = data[sensor][trial]

    # Get array data
    x_array = epochs.get_data()

    if config['n_class'] == 3:
        # 3-class encoding
        if trial == 'HighFine':
            category = [1, 0, 0]
        elif trial == 'LowGross':
            category = [0, 1, 0]
        elif (trial == 'HighGross') or (trial == 'LowFine'):
            category = [0, 0, 1]

    elif config['n_class'] == 4:
        # 4-class encoding
        if trial == 'HighFine':
            category = [1, 0, 0, 0]
        elif trial == 'LowGross':
            category = [0, 1, 0, 0]
        elif trial == 'HighGross':
            category = [0, 0, 1, 0]
        elif trial == 'LowFine':
            category = [0, 0, 0, 1]

    # In order to accomodate testing
    try:
        y_array = one_hot_encode(x_array.shape[0], category)
    except ImportError:
        y_array = np.zeros((x_array.shape[0], config['n_class']))

    return x_array, y_array


def convert_test_trials_to_array(subject, emg_path, pb_path, trial, config):
    """Label the data and convert it to array.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trial : str
        Trail e.g. HighFine, LowGross.
    config : yaml
        The configuration file.

    Returns
    -------
    emg_array
        An array of EMG data
    pos_array
        An array of PB data
    y_array
        An array of true labels
    """
    
    if trial in config['comb_trials']:
        # load emg data 
        # emg_path    = str(Path(__file__).parents[2] / config['epoch_emg_data'])
        data        = dd.io.load(emg_path, group='/' + 'subject_' + subject)
        epochs_emg  = data['EMG'][trial]

        # load pb data
        # pb_path  = str(Path(__file__).parents[2] / config['epoch_PB_data'])
        data  = dd.io.load(pb_path, group='/' + 'subject_' + subject)
        epochs_pb   = data['PB'][trial]
        
        # Get array data
        emg_array = epochs_emg.get_data()
        pb_array  = epochs_pb.get_data()

        # match the PB and EMG data size    
        array_len = np.min([emg_array.shape[0], pb_array.shape[0]])
        emg_array = emg_array[:array_len, :, :]
        pb_array  = pb_array[:array_len, :, :]

        # Label the combined task based on the position information
        pos = pb_array[:,2:4,:].mean(axis=2)
        pos_ind_gross = pos[:,1] < config['comb_task_pos_limits'][0]
        pos_ind_fine  = pos[:,1] > config['comb_task_pos_limits'][1]
        
        # Remove the data from the Gross-Fine transition region as they cannot be labelled
        pos_ind_del   = ((pos[:,1] > config['comb_task_pos_limits'][0]) & (pos[:,1] < config['comb_task_pos_limits'][1]))
        ind           = np.arange(0,len(pos_ind_del))
        pos_ind_del   = ind[pos_ind_del]

        # assign the class labels
        y_array = np.zeros((emg_array.shape[0], config['n_class']))
        if config['n_class'] == 3:
            if trial == 'HighComb':
                category_gross = np.array([0, 0, 1]).reshape(1,config['n_class']) 
                category_fine  = np.array([1, 0, 0]).reshape(1,config['n_class']) 
            elif trial == 'LowComb':
                category_gross = np.array([0, 1, 0]).reshape(1,config['n_class']) 
                category_fine  = np.array([0, 0, 1]).reshape(1,config['n_class']) 
        
        elif config['n_class'] == 4:
            if trial == 'HighComb':
                category_gross = np.array([0, 0, 1, 0]).reshape(1,config['n_class']) 
                category_fine  = np.array([1, 0, 0, 0]).reshape(1,config['n_class']) 
            elif trial == 'LowComb':
                category_gross = np.array([0, 1, 0, 0]).reshape(1,config['n_class']) 
                category_fine  = np.array([0, 0, 0, 1]).reshape(1,config['n_class']) 
        
        y_array[pos_ind_gross, :] = category_gross + y_array[pos_ind_gross, :]
        y_array[pos_ind_fine, :]  = category_fine  + y_array[pos_ind_fine, :]

        # delete the elements from the transition region
        emg_array = np.delete(emg_array, pos_ind_del, axis=0)
        pb_array = np.delete(pb_array, pos_ind_del, axis=0)
        y_array = np.delete(y_array, pos_ind_del, axis=0)

        # y = np.dot(y_array,np.array(np.arange(1, config['n_class']+1)))        
        # plots to check if the labels are assigned properly
        # fig = plt.figure()
        # ax = Axes3D(fig)
        
        # ax.scatter(pos[pos_ind_gross, 0], pos[pos_ind_gross, 1], y[pos_ind_gross], 'r.')
        # ax.scatter(pos[pos_ind_fine > 0.125, 0], pos[pos_ind_fine, 1], y[pos_ind_fine], 'g.')
        # plt.show()
    else: 
        emg_array, y_array = convert_to_array(subject, trial, emg_path, 'EMG', config)
        pb_array, y_array = convert_to_array(subject, trial, pb_path, 'PB', config)

    return emg_array, pb_array, y_array


def clean_epoch_data(subjects, trials, sensor, config):
    """Create feature dataset for all subjects.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials
    sensor : str
        Selection of data from sensor: 'EMG' or 'PB'
    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:
        # Initialize for each subject
        x = []
        y_temp = []

        if subject not in config['test_subjects']:
            if (sensor == 'EMG'):
                path = str(Path(__file__).parents[2] / config['epoch_emg_data'])
            elif (sensor == 'PB'):
                path = str(Path(__file__).parents[2] / config['epoch_PB_data'])

            # (set(list_a) ^ set(list_b)) or (set(list_a) - set(list_b))) gives the elements in list_a that are not in list_b
            for trial in (set(config['trials']) - set(config['comb_trials'])):
                # Concatenate the data corresponding to all trial types           
                x_array, y_array = convert_to_array(subject, trial, path, sensor, config)
                x_temp.append(x_array)
                y_temp.append(y_array)

        elif (config['pool_comb_task_data']):            
            emg_path = str(Path(__file__).parents[2] / config['epoch_emg_data'])
            pb_path = str(Path(__file__).parents[2] / config['epoch_PB_data'])

            # for trial in config['comb_trials']: 
            for trial in config['trials']:            
                emg_array, pb_array, y_array = convert_test_trials_to_array(subject, emg_path, pb_path, trial, config)
                
                if (sensor == 'PB'):
                    x_array = pb_array
                elif (sensor == 'EMG'):
                    x_array = emg_array

                x_temp.append(x_array)
                y_temp.append(y_array)
        else:            
            emg_path = str(Path(__file__).parents[2] / config['epoch_emg_data'])
            pb_path = str(Path(__file__).parents[2] / config['epoch_PB_data'])

            # for trial in config['comb_trials']: 
            for trial in (set(config['trials']) - set(config['comb_trials'])):            
                emg_array, pb_array, y_array = convert_test_trials_to_array(subject, emg_path, pb_path, trial, config)
                
                if (sensor == 'PB'):
                    x_array = pb_array
                elif (sensor == 'EMG'):
                    x_array = emg_array

                x_temp.append(x_array)
                y_temp.append(y_array)

        # Convert to array
        x_temp = np.concatenate(x_temp, axis=0)
        y_temp = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['features'] = np.float32(x_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)
    return features_dataset


def clean_combined_data(subjects, trials, n_class, config):
    """Create combined EMG and Force feature set for all subjects.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials
    n_class : int
        label the data into these many classes

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:
        # Initialise for each subject
        EMG_temp = []
        PB_temp  = []
        y_temp   = []

        if subject not in config['test_subjects']:
            for trial in (set(trials) - set(config['comb_trials'])):
                # if trial not in config['comb_trials']:
                path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
                path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

                # Concatenate the data corresponding to all trials types
                EMG_array, y_array = convert_to_array(subject, trial, path1, 'EMG', config)
                
                data = dd.io.load(path2, group='/' + 'subject_' + subject)
                epochs = data['PB'][trial]
                PB_array = epochs.get_data()

                # match the PB and EMG data size    
                array_len = np.min([EMG_array.shape[0], PB_array.shape[0]])
                EMG_array = EMG_array[:array_len, :, :]
                PB_array  = PB_array[:array_len, :, :] 
                y_array  = y_array[:array_len, :] 

                EMG_temp.append(EMG_array)
                PB_temp.append(PB_array)
                y_temp.append(y_array)
        else:
            if config['test_all_trials']:
                use_trials = config['trials']
            elif config['test_comb_trials']:
                use_trials = config['comb_trials']
            else:
                use_trials = set(config['trials']) - set(config['comb_trials'])

            for trial in use_trials:
                path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
                path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

                # Concatenate the data corresponding to all trials types
                EMG_array, PB_array, y_array = convert_test_trials_to_array(subject, path1, path2, trial, config)
                
                EMG_temp.append(EMG_array)
                PB_temp.append(PB_array)
                y_temp.append(y_array)

        # Convert to array
        EMG_temp = np.concatenate(EMG_temp, axis=0)
        PB_temp  = np.concatenate(PB_temp, axis=0)
        y_temp   = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['EMG'] = np.float32(EMG_temp)
        features_dataset['subject_' + subject]['PB'] = np.float32(PB_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)

    return features_dataset


def clean_combined_data_for_fatigue_study(subjects, trials, n_class, config):
    """Create combined EMG and Force feature set for all subjects.
    This contains all the data from the subjects Eg. LowComb, HighComb etc

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials
    n_class : int
        label the data into these many classes

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:
        # Initialise for each subject
        EMG_temp = []
        PB_temp  = []
        y_temp   = []

        if subject not in config['test_subjects']:
            for trial in trials:
                # if trial not in config['comb_trials']:
                path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
                path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

                # Concatenate the data corresponding to all trials types
                EMG_array, y_array = convert_to_array(subject, trial, path1, 'EMG', config)
                
                data = dd.io.load(path2, group='/' + 'subject_' + subject)
                epochs = data['PB'][trial]
                PB_array = epochs.get_data()

                # match the PB and EMG data size    
                array_len = np.min([EMG_array.shape[0], PB_array.shape[0]])
                EMG_array = EMG_array[:array_len, :, :]
                PB_array  = PB_array[:array_len, :, :] 
                y_array  = y_array[:array_len, :] 

                EMG_temp.append(EMG_array)
                PB_temp.append(PB_array)
                y_temp.append(y_array)
        else:

            for trial in trials:
                path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
                path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

                # Concatenate the data corresponding to all trials types
                EMG_array, PB_array, y_array = convert_test_trials_to_array(subject, path1, path2, trial, config)
                
                EMG_temp.append(EMG_array)
                PB_temp.append(PB_array)
                y_temp.append(y_array)

        # Convert to array
        EMG_temp = np.concatenate(EMG_temp, axis=0)
        PB_temp  = np.concatenate(PB_temp, axis=0)
        y_temp   = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['EMG'] = np.float32(EMG_temp)
        features_dataset['subject_' + subject]['PB'] = np.float32(PB_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)

    return features_dataset


def clean_intersession_test_data(subjects, trials, n_class, config):
    """Create combined EMG and Force feature set for all subjects.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials
    n_class : int
        label the data into these many classes

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:
        # Initialise for each subject
        EMG_temp = []
        PB_temp  = []
        y_temp   = []

        if subject in config['test_subjects']:
            # for trial in config['comb_trials']:
            for trial in config['trials']:
                path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
                path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

                # Concatenate the data corresponding to all trials types
                EMG_array, PB_array, y_array = convert_test_trials_to_array(subject, path1, path2, trial, config)
                
                EMG_temp.append(EMG_array)
                PB_temp.append(PB_array)
                y_temp.append(y_array)

        # Convert to array
        EMG_temp = np.concatenate(EMG_temp, axis=0)
        PB_temp  = np.concatenate(PB_temp, axis=0)
        y_temp   = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['EMG'] = np.float32(EMG_temp)
        features_dataset['subject_' + subject]['PB'] = np.float32(PB_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)

    return features_dataset


# The following function are for the prediction correction using the history of points
def clean_correction_data(subjects, trials, n_class, config):
    """Create combined EMG and Force feature set for all subjects for the correction algorithm 

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials
    n_class : int
        label the data into these many classes

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:

        loaded_data = collections.defaultdict()
        if subject not in config['test_subjects']: 
            for trial in (set(trials) - set(config['comb_trials'])):
                created_data = collections.defaultdict()

                # if trial not in config['comb_trials']:
                path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
                path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

                # Concatenate the data corresponding to all trials types
                EMG_array, y_array = convert_to_array(subject, trial, path1, 'EMG', config)
                
                data = dd.io.load(path2, group='/' + 'subject_' + subject)
                epochs = data['PB'][trial]
                PB_array = epochs.get_data()

                # match the PB and EMG data size    
                array_len = np.min([EMG_array.shape[0], PB_array.shape[0]])
                EMG_array = EMG_array[:array_len, :, :]
                PB_array  = PB_array[:array_len, :, :] 
                y_array   = y_array[:array_len, :] 

                created_data['EMG'] = EMG_array
                created_data['PB'] = PB_array
                created_data['labels'] = y_array

                loaded_data[trial] = created_data
        else:
            if config['test_all_trials']:
                use_trials = config['trials']
            elif config['test_comb_trials']:
                use_trials = config['comb_trials']
            else:
                use_trials = set(config['trials']) - set(config['comb_trials'])

            for trial in use_trials:
                created_data = collections.defaultdict()

                path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
                path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

                # Concatenate the data corresponding to all trials types
                EMG_array, PB_array, y_array = convert_test_trials_to_array(subject, path1, path2, trial, config)
                
                created_data['EMG'] = EMG_array
                created_data['PB'] = PB_array
                created_data['labels'] = y_array 

                loaded_data[trial] = created_data

        # Append to the big dataset
        features_dataset['subject_' + subject] = loaded_data

    return features_dataset


def find_balancing_cutoff_index(data, trials):
    """find the length of the shortest trial to balance the other experimental trials accordingly
    
    Parameters
    ----------
    data : dictionary
        data dictionary consisting of trials from each subject
    trials : list
        list of experimental trials provided in config.yml
    
    Returns
    -------
    min_ind : int
        length of the shortest trial across the trials
    """

    ind_list = [data[trial]['labels'].shape[0] for trial in trials]
    min_ind = min(ind_list)

    # making the index odd i.e length of vector even for balancing the data in the next step  
    if (min_ind % 2) == 0:
        min_ind -= 1

    return min_ind
        

def balance_correction_data(data, trials, subjects, config, balance=True):
    """balance the data for each subject to pass it on to the correction algorithm

    Parameter
    ----------
    data : dictionary
        dictionary of data created using "clean_data_for_correction_alogorithm" function
    trials : list
        A list of differet trials
    subject : list
        list of subject IDs e.g. 0001.
    config : dictionary
        dictionary of parameters loaded from the config.yml file
    balance : bool
        balance the data if True, leave the data as is if False

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """

    for subject in subjects:
        min_ind = find_balancing_cutoff_index(data['subject_'+subject], trials)
        
        for trial in trials:
            if (config['n_class'] == 3 and trial == 'HighGross') or (config['n_class'] == 3 and trial == 'LowFine'):
                ind = int((min_ind - 1) / 2)
            else:
                ind = min_ind
            data['subject_' + subject][trial]['EMG']    = data['subject_' + subject][trial]['EMG'][:ind+1, :, :]
            data['subject_' + subject][trial]['PB']     = data['subject_' + subject][trial]['PB'][:ind+1, :]
            data['subject_' + subject][trial]['labels'] = data['subject_' + subject][trial]['labels'][:ind+1, :]

    return data


def pool_correction_data(data, subjects, trials, config):
    """balance the data for each subject to pass it on to the correction algorithm

    Parameter
    ----------
    data : dictionary
        dictionary of data created using "clean_data_for_correction_alogorithm" function
    subject : str
        String of subject ID e.g. 0001. look at the config.yml file for more information
    trials : list
        A list of differet trials

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialise for each subject
    EMG = []
    PB  = []
    y   = []

    for subject in subjects:
        for trial in trials:
            EMG.append(data['subject_' + subject][trial]['EMG'])
            PB.append( data['subject_' + subject][trial]['PB'])
            y.append(  data['subject_' + subject][trial]['labels'])

    # Convert to array
    EMG = np.concatenate(EMG, axis=0)
    PB  = np.concatenate(PB, axis=0)
    y   = np.concatenate(y, axis=0)

    return EMG, PB, y

