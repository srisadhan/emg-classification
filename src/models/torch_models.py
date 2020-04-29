import numpy as np
import torch
import torch.nn as nn
from torchnet.logger import VisdomPlotLogger
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


def train_correction_network(network, config, data):
    """train the ShallowCorrectionNet on the same training data used to 
       train the SVM classifier in order to correct the predictions.

    Parameters
    ----------
    network : neural network
        A pytorch network.
    config : yaml
        The configuration file.
    data : dict
        A dictionary comprising of training, validation, and testing data
    
    """
    # Device to train the model cpu or gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device being used:', device)

    # An instance of model
    model = network(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    # visual logger 
    visual_logger1 = visual_log('Task type classification using self-correction')
    visual_logger2 = VisdomPlotLogger('line', 
                                    opts=dict(xlabel='Epochs',
                                        ylabel='Error',
                                        title='Error plot'))
    accuracy_log  = []

    for epoch in range(config['NUM_EPOCHS']):
        for x_batch, y_batch in data['training']:
            
            x_batch = x_batch.to(device)
            # y_batch = y_batch.to(device) # use this while using MSELoss() otherwise use below
            y_batch = (torch.max(y_batch, dim=1)[1]).to(device) #convert labels from one hot encoding to normal

            # Forward propagation
            net_out = model(x_batch)
            loss    = criterion(net_out, y_batch)

            # Back propagation 
            optimizer.zero_grad()  # For batch gradient optimisation
            loss.backward()
            optimizer.step()
        
        accuracy = classification_accuracy(model, data)
        accuracy_log.append(accuracy)

        # log the accuracies
        visual_logger1.log(epoch, [accuracy[0], accuracy[1], accuracy[2]])

        # log the errors
        visual_logger2.log(epoch, loss.item())

    # Add loss function info to parameter.
    model_info = create_model_info(config, str(criterion),
                                   np.array(accuracy_log))

    return model, model_info

    #FIXME: is the data balanced?
    #FIXME: can the median yield better results
    #FIXME: plot the error
