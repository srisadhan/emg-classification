import yaml
from pathlib import Path

from data.clean_data import clean_epoch_data
from data.create_data import create_emg_data, create_emg_epoch
from datasets.riemann_datasets import subject_pooled_data, train_test_data
from datasets.torch_datasets import pooled_data_iterator

from models.riemann_models import (svm_tangent_space_classifier,
                                   svm_tangent_space_cross_validate,
                                   svm_tangent_space_prediction)
from models.torch_models import train_torch_model
from models.torch_networks import ShallowERPNet

from visualization.visualise import plot_average_model_accuracy

from utils import (skip_run, save_data, save_trained_pytorch_model)

# The configuration file
config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)

with skip_run('skip', 'create_emg_data') as check, check():
    data = create_emg_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['raw_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'create_epoch_data') as check, check():
    data = create_emg_epoch(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['epoch_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'clean_epoch_data') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['clean_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'pooled_data_svm') as check, check():
    # Load main data
    features, labels, leave_tags = subject_pooled_data(config)

    # Get the data
    data = train_test_data(features, labels, leave_tags, config)

    # Train the classifier and predict on test data
    clf = svm_tangent_space_classifier(data['train_x'], data['train_y'])
    svm_tangent_space_prediction(clf, data['test_x'], data['test_y'])

with skip_run('skip', 'pooled_data_svm_cross_validated') as check, check():
    # Load main data
    features, labels, leave_tags = subject_pooled_data(config)

    # Get the data
    data = train_test_data(features, labels, leave_tags, config)
    svm_tangent_space_cross_validate(data)

with skip_run('skip', 'pooled_data_torch') as check, check():
    dataset = pooled_data_iterator(config)
    model, model_info = train_torch_model(ShallowERPNet, config, dataset)
    path = Path(__file__).parents[1] / config['trained_model_path']
    save_path = str(path)
    save_trained_pytorch_model(model, model_info, save_path, save_model=False)

with skip_run('skip', 'average_accuracy') as check, check():
    plot_average_model_accuracy('experiment_0', config)
