import torch
import torch.nn as nn


class ShallowERPNet(nn.Module):
    """Convolution neural network class for eeg classification.

    Parameters
    ----------
    OUTPUT : int
        Number of classes.

    Attributes
    ----------
    net_1 : pytorch Sequential
        Convolution neural network class for eeg classification.
    pool : pytorch pooling
        Pooling layer.
    net_2 : pytorch Sequential
        Classification convolution layer.

    """
    def __init__(self, OUTPUT, config):
        super(ShallowERPNet, self).__init__()
        # Configuration of EMGsignals
        self.epoch_length = config['epoch_length']
        self.s_freq = config['sfreq']
        self.n_electrodes = config['n_electrodes']

        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(1, 15), stride=1),
            nn.Conv2d(20, 20, kernel_size=(8, 8), stride=1),
            nn.BatchNorm2d(20, momentum=0.1, affine=True))

        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))

        self.dropout = nn.Dropout(p=config['DROP_OUT'])

        self.net_2 = nn.Sequential(
            nn.Conv2d(20, OUTPUT, kernel_size=(1, 7), stride=1),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x[:, None, :, :]  # Add the extra dimension
        out = self.net_1(x)
        out = out * out
        out = self.pool(out)
        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.dropout(out)
        out = self.net_2(out)
        out = torch.squeeze(out)

        return out
