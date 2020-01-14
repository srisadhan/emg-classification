import collections
from pathlib import Path
from data.clean_data import convert_to_array
import numpy as np
import sys 
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

def extract_passivity_index(subjects, trials, sensor, config):
    """extract the passivity index using the force and velocity for all subjects.

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
        # Initialise for each subject
        PI_temp = []
        y_temp = []
        for trial in trials:
            path = str(Path(__file__).parents[2] / config['epoch_PB_data'])

            # Concatenate the data corresponding to all trials types
            x_array, y_array = convert_to_array(subject, trial, path, sensor, config)

            # plt.figure()
            # for i in x_array:
                # to verify if the force direction match with the position
                # for j in range(0,40,10):
                #     plt.quiver(i[2,j], i[3,j], i[0,j], i[1,j])
                #     plt.pause(.0001)
            # plt.show()
            
            # Calculating the passivity index
            PI = np.zeros((x_array.shape[0],2))
            for i in range(0, x_array.shape[0]):
                force = x_array[i,0:2,:]
                pos   = x_array[i,2:,:]

                # calculating velocity from the position
                vel   = np.diff(pos, axis=1) * config['sfreq2']
                # drop the first value of force to match the dimension with velocity
                force = force[:,1:]

                # variation of force and velocity with respect to the initial value
                force_diff = force - force[:,0].reshape(2,1)
                vel_diff   = vel - vel[:,0].reshape(2,1)
                PI[i,:]    = config['sfreq2'] * np.sum(np.multiply(force_diff, vel_diff), axis=1).reshape(1,2)

            PI_temp.append(PI)
            y_temp.append(y_array)

        # Convert to array
        PI_temp = np.concatenate(PI_temp, axis=0)
        y_temp = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['PI'] = np.float32(PI_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)

    return features_dataset


def pool_force_features(features_dataset, config):
    """pool the extracted force features.

    Parameter
    ----------
    features_dataset : dictionary
        a dictionary containing force features and the corresponding labels
    config : yaml
    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    subjects = config['subjects']

    # Empty list
    x = []
    y = []
    for subject in subjects:
        x_temp = features_dataset['subject_' + subject]['PI'] 
        y_temp = features_dataset['subject_' + subject]['labels']

        x.append(x_temp)
        y.append(y_temp)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :]
    labels = y[rus.sample_indices_, :]

    return features, labels

 