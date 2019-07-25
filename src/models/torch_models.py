import numpy as np
import torch
import torch.nn as nn

from .utils import (visual_log, classification_accuracy, create_model_info,
                    weights_init)


def train_torch_model(network, config, data_iterator, new_weights=False):
    """Main function to run the optimization.

    Parameters
    ----------
    network : class
        A pytorch network class.
    config : yaml
        The configuration file.
    data_iterator : dict
        A data iterator with training, validation, and testing data
    new_weights : bool
        Whether to use new weight initialization instead of default.

    Returns
    -------
    pytorch model
        A trained pytroch model.

    """
    # Device to train the model cpu or gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device being used:', device)

    # An instance of model
    model = network(config['OUTPUT'], config).to(device)
    if new_weights:
        model.apply(weights_init)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['LEARNING_RATE'])

    # Visual logger
    visual_logger = visual_log('Task type classification')
    accuracy_log = []
    for epoch in range(config['NUM_EPOCHS']):
        for x_batch, y_batch in data_iterator['training']:
            # Send the input and labels to gpu
            x_batch = x_batch.to(device)
            y_batch = (torch.max(y_batch, dim=1)[1]).to(device)

            # Forward pass
            out_put = model(x_batch)
            loss = criterion(out_put, y_batch)

            # Backward and optimize
            optimizer.zero_grad()  # For batch gradient optimisation
            loss.backward()
            optimizer.step()

        accuracy = classification_accuracy(model, data_iterator)
        accuracy_log.append(accuracy)
        visual_logger.log(epoch, [accuracy[0], accuracy[1], accuracy[2]])

    # Add loss function info to parameter.
    model_info = create_model_info(config, str(criterion),
                                   np.array(accuracy_log))

    return model, model_info
