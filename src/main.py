import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from data.clean_data import clean_epoch_data
from data.create_data import (create_emg_data, create_emg_epoch,
                              create_robot_dataframe)
from datasets.riemann_datasets import subject_pooled_data, train_test_data
from datasets.torch_datasets import pooled_data_iterator
from datasets.statistics_dataset import matlab_dataframe

from models.riemann_models import (svm_tangent_space_classifier,
                                   svm_tangent_space_cross_validate,
                                   svm_tangent_space_prediction)
from models.statistical_models import mixed_effect_model
from models.torch_models import train_torch_model
from models.torch_networks import ShallowERPNet

from visualization.visualise import (plot_average_model_accuracy, plot_bar)

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

with skip_run('skip', 'create_statistics_dataframe') as check, check():
    data = create_robot_dataframe(config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['statistics_dataframe']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'statistical_analysis') as check, check():
    dataframe = matlab_dataframe(config)

    vars = ['task + damping', 'task * damping']

    # # Perform for total force
    # for var in vars:
    #     md_task = mixed_effect_model(dataframe,
    #                                  dependent='total_force',
    #                                  independent=var)
    # Perform for velocity
    for var in vars:
        print(var)
        md_task = mixed_effect_model(dataframe,
                                     dependent='velocity',
                                     independent=var)

with skip_run('skip', 'svm_pooled_data') as check, check():
    # Load main data
    features, labels, leave_tags = subject_pooled_data(config)

    # Get the data
    data = train_test_data(features, labels, leave_tags, config)

    # Train the classifier and predict on test data
    clf = svm_tangent_space_classifier(data['train_x'], data['train_y'])
    svm_tangent_space_prediction(clf, data['test_x'], data['test_y'])

with skip_run('skip', 'svm_cross_validated_pooled_data') as check, check():
    # Load main data
    features, labels, leave_tags = subject_pooled_data(config)

    # Get the data
    data = train_test_data(features, labels, leave_tags, config)
    svm_tangent_space_cross_validate(data)

with skip_run('skip', 'torch_pooled_data') as check, check():
    dataset = pooled_data_iterator(config)
    model, model_info = train_torch_model(ShallowERPNet, config, dataset)
    path = Path(__file__).parents[1] / config['trained_model_path']
    save_path = str(path)
    save_trained_pytorch_model(model, model_info, save_path, save_model=False)

with skip_run('skip', 'average_accuracy') as check, check():
    plot_average_model_accuracy('experiment_0', config)

with skip_run('skip', 'bar_plot') as check, check():
    # Get the data
    dataframe = matlab_dataframe(config)

    plt.subplots(figsize=(7, 4))
    sns.set(font_scale=1.2)

    # Force
    plt.subplot(1, 2, 1)
    dependent = 'task'
    independent = 'total_force'
    plot_bar(config, dataframe, independent, dependent)

    # Velocity
    plt.subplot(1, 2, 2)
    dependent = 'task'
    independent = 'velocity'
    plot_bar(config, dataframe, independent, dependent)

    plt.tight_layout()
    plt.show()
