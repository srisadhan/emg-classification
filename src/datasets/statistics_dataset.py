from pathlib import Path

import deepdish as dd
import pandas as pd


def subject_pooled_dataframe(config):
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
    path = path = str(
        Path(__file__).parents[2] / config['statistics_dataframe'])
    dataframe = dd.io.load(path)

    return dataframe


def matlab_dataframe(config):
    """Get subject independent data (pooled data) from external matlab file.

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
    path = path = str(Path(__file__).parents[2] / config['matlab_dataframe'])
    dataframe = pd.read_csv(path)

    return dataframe
