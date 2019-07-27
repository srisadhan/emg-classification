from pathlib import Path

import matplotlib.pyplot as plt


def get_model_path(experiment, model_number):
    """Get all the trained model paths from experiment.

    Parameters
    ----------
    experiment : str
        Which experiment trained models to load.

    Returns
    -------
    model path and model info path

    """

    read_path = str(Path(__file__).parents[2]) + '/models/' + experiment
    with open(read_path + '/time.txt', "r+") as f:
        trained_model = f.readlines()[model_number]
    model_time = trained_model.splitlines()[0]  # remove "\n"
    model_path = str(
        Path(__file__).parents[2]
    ) + '/models/' + experiment + '/model_' + model_time + '.pth'
    model_info_path = str(
        Path(__file__).parents[2]
    ) + '/models/' + experiment + '/model_info_' + model_time + '.pth'

    return model_path, model_info_path


def figure_asthetics(ax):
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """

    ax.yaxis.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return None


def annotate_significance(x1, x2, y, p):
    """Add significance annotations over a plot.

    Parameters
    ----------
    x1 : float
        x position of factor 1.
    x2 : float
        x position of factor 2.
    y : float
        Outcome variable.

    Returns
    -------
    None

    """
    h = y * 0.025
    star = []
    if p < 0.001:
        star = "***"
    elif p < 0.01:
        star = "**"
    if star:
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
        plt.text((x1 + x2) * .5,
                 y,
                 star,
                 ha='center',
                 va='bottom',
                 color='k',
                 size=20)

    return None
