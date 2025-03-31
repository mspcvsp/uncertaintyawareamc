"""
Automatic Modulation Classification (AMC) Bayesian neural network

References
----------
[1] V. -C. Luu, J. Park and J. -P. Hong, "Uncertainty-Aware Incremental
Automatic Modulation Classification With Bayesian Neural Network," in
IEEE Internet of Things Journal, vol. 11, no. 13, pp. 24300-24309,
1 July 1, 2024, doi: 10.1109/JIOT.2024.3390038.
"""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from .net_utils import init_weights


class ResidualBlock(nn.Module):
    """
    Residual network building block class
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        """
        Residual block class object constructor

        Parameters
        ----------
        self: ResidualBlock
            ResidualBlock class object reference

        kwargs: dict
            Stores optional parameters

        Returns
        -------
        self: ResidualBlock
            ResidualBlock class object reference
        """
        super().__init__()

        """
        conv, 1, N, linear
        """
        self.conv1 =\
            nn.Sequential(
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size=1),
                nn.BatchNorm1d(out_channels))

        """
        conv, 3, N, ReLU
        conv, 3, N, linear
        """
        self.conv2 =\
            nn.Sequential(
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels))

        """
        conv, 3, N, ReLU
        conv, 3, N, linear
        """
        self.conv3 =\
            nn.Sequential(
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels))

        """
        Shortcut layers
        """
        self.shortcut1 =\
            nn.Sequential(
                    nn.Conv1d(out_channels,
                              out_channels,
                              kernel_size=1,
                              bias=False),
                    nn.BatchNorm1d(out_channels))

        self.shortcut2 =\
            nn.Sequential(
                    nn.Conv1d(out_channels,
                              out_channels,
                              kernel_size=1,
                              bias=False),
                    nn.BatchNorm1d(out_channels))

        self.max_pool =\
            nn.MaxPool1d(kernel_size=3,
                         stride=2,
                         padding=1)

        self.apply(init_weights)

    def forward(self,
                x):
        """
        Implements the foward pass of a residual network building block

        Parameters
        ----------
        self: ResidualBlock
            ResidualBlock class object reference

        x: torch.Tensor
            Input tensor

        Returns
        -------
        y: torch.Tensor
            Output tensor
        """
        layer1_out = self.conv1(x.clone())
        layer2_out = self.conv2(layer1_out)
        sc1_out = self.shortcut1(layer1_out)
        layer3_in = layer2_out + sc1_out

        layer3_out = self.conv2(layer3_in)
        sc2_out = self.shortcut2(layer3_in)

        return self.max_pool(layer3_out + sc2_out)


class IQDataEmbedder(nn.Module):
    """
    """
    def __init__(self):

        super().__init__()

        """
        Layer #1
        ---------
        conv 3, 64
        ReLU
        max pool / 2
        """
        self.conv1 =\
            nn.Sequential(
                nn.Conv1d(in_channels=2,
                          out_channels=64,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3,
                             stride=2,
                             padding=1))
        
        """
        Layer #2
        ---------
        conv 3, 64
        ReLU
        max pool / 2
        """
        self.conv2 =\
            nn.Sequential(
                nn.Conv1d(in_channels=64,
                          out_channels=128,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3,
                             stride=2,
                             padding=1))
        
        """
        Layer #3
        ---------
        conv 3, 128
        ReLU
        max pool / 2
        """
        self.conv3 =\
            nn.Sequential(
                nn.Conv1d(in_channels=128,
                          out_channels=256,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3,
                             stride=2,
                             padding=1))
        
        """
        Layer #4
        ---------
        conv 3, 128
        ReLU
        max pool / 2
        """
        self.conv4 =\
            nn.Sequential(
                nn.Conv1d(in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3,
                             stride=2,
                             padding=1))

        self.lstm = nn.LSTM(256,
                            64,
                            num_layers=2,
                            batch_first=True)

        self.apply(init_weights)

    def forward(self,
                x):

        layer_out = self.conv1(x)
        layer_out = self.conv2(layer_out)
        layer_out = self.conv3(layer_out)
        layer_out = self.conv4(layer_out)

        layer_out = torch.permute(layer_out, (0, 2, 1))
        layer_out, _ = self.lstm(layer_out)
        layer_out = torch.permute(layer_out, (0, 2, 1))

        return torch.flatten(layer_out, start_dim=1)


class FrequentistClassifier(nn.Module):

    def __init__(self,
                 **kwargs):

        super().__init__()

        self.num_classes = kwargs.get("num_classes", 18)

        self.fc_layers =\
            nn.Sequential(nn.Linear(8192, 2048),
                          nn.Dropout(0.5),
                          nn.BatchNorm1d(2048),
                          nn.ReLU(),
                          nn.Linear(2048, 512),
                          nn.Dropout(0.5),
                          nn.BatchNorm1d(512),
                          nn.ReLU(),
                          nn.Linear(512, 18))

        self.apply(init_weights)

    def forward(self,
                x):

        return self.fc_layers(x)


class AutoModClassifier(pl.LightningModule):
    """
    """
    def __init__(self,
                 **kwargs):
        """
        """
        super(AutoModClassifier,
              self).__init__()

        self.optim_params = dict()

        self.optim_params["lr"] =\
            kwargs.get("lr", 1E-3)

        self.iq_embed = IQDataEmbedder()
        self.modclf = FrequentistClassifier(**kwargs)

        self.loss = nn.CrossEntropyLoss()

        self.train_accuracy =\
            tm.Accuracy(task="multiclass",
                        num_classes=self.modclf.num_classes)

        self.val_accuracy =\
            tm.Accuracy(task="multiclass",
                        num_classes=self.modclf.num_classes)

        self.test_accuracy =\
            tm.Accuracy(task="multiclass",
                        num_classes=self.modclf.num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.apply(init_weights)

    def forward(self,
                x,
                applySoftmax=True):
        """

        Parameters
        ----------
        self: AutoModClassifier
            Automatic modulation classifier class object reference

        x: torch.Tensor
            [Batch size] x 2048 sample (I/Q) sample vector

        applySoftmax: bool (Optional)
            Boolean that controls whether to apply softmax to linear
            layers output. This input should be set to false during
            training to be compatible with CrossEntropy loss

        Returns
        -------
        [batch size x number of classes] tensor that stores the
            predicted classes for each network input
        """
        layer_out = self.iq_embed(x)
        net_output = self.modclf(layer_out)

        if applySoftmax:
            net_output = self.softmax(net_output)

        return net_output

    def configure_optimizers(self):
        """
        Configures optimizer function(s)

        Parameters
        ----------
        self: AutoModClassifier
            Automatic modulation classifier class object reference

        Returns
        ----------
        self: AutoModClassifier
            Automatic modulation classifier class object reference
        """
        optimizer = optim.Adam(params=self.parameters(),
                               lr=self.optim_params.get("lr", 1E-3))

        return optimizer

    def training_step(self,
                      batch,
                      batch_idx):
        """
        Implements a Pytorch Lightning module training step

        Parameters
        ----------
        self: AutoModClassifier
            AutoModClassifier class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        train_loss: float
            Batch training loss
        """
        batch_data, modeordenc, _ = batch

        logits = self.forward(batch_data,
                              applySoftmax=False)

        target = self.init_target(modeordenc)

        train_loss = self.loss(logits, target)

        self.train_accuracy(self.softmax(logits),
                            modeordenc)

        self.log("train_loss",
                 train_loss,
                 prog_bar=True)

        self.log("train_accuracy",
                 self.train_accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return train_loss

    def predict_step(self,
                     batch,
                     batch_idx):
        """
        Implements a Pytorch Lightning module prediction step

        Parameters
        ----------
        self: AutoModClassifier
            Automatic modulation classifier class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        predictions : Tensor
            [Batch size x number of classes] tensor that stores batch
            class predicitions

        modeordenc : Tensor
            [Batch size x 1] tensor that stores the ordinal encoded
            modulation mode

        snrordenc : Tensor
            [Batch size x 1] tensor that stores the ordinal encoded
            Signal-to-Noise Ratio (SNR)
        """
        batch_data, modeordenc, snrordenc = batch

        predictions = self.forward(batch_data)

        return predictions, modeordenc, snrordenc

    def validation_step(self,
                        batch,
                        batch_idx):
        """
        Implements a Pytorch Lightning module validation step

        Parameters
        ----------
        self: AutoModClassifier
            AutoModClassifier class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        None
        """
        batch_data, modeordenc, _ = batch

        logits = self.forward(batch_data,
                              applySoftmax=False)

        target = self.init_target(modeordenc)

        self.val_accuracy(self.softmax(logits),
                          modeordenc)

        self.log("validation_loss",
                 self.loss(logits, target),
                 prog_bar=True)

        self.log("val_acc",
                 self.val_accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def test_step(self,
                  batch,
                  batch_idx):
        """
        Implements a Pytorch Lightning module test step

        Parameters
        ----------
        self: AutoModClassifier
            AutoModClassifier class object reference

        batch: tuple
            Tuple that stores batch data, ordinal encoded modulation mode &
            ordinal encoded Signal-to-Noise Ratio (SNR)

        batch_idx: integer
            Batch index

        Returns
        -------
        None
        """
        batch_data, modeordenc, _ = batch

        logits = self.forward(batch_data,
                              applySoftmax=False)

        target = self.init_target(modeordenc)

        self.test_accuracy(self.softmax(logits),
                           modeordenc)

        self.log("test_loss",
                 self.loss(logits, target),
                 prog_bar=True)

        self.log("test_acc",
                 self.test_accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def init_target(self,
                    modeordenc):
        """
        Initializes a tensor that stores target predicted class confidence
        (for the cross entropy loss) given a batch's ordinal encoded
        modulation mode

        Parameters:
        ----------
        self: AutoModClassifier
            AutoModClassifier class object reference

        modeordenc : Tensor
            [Batch size x 1] tensor that stores the ordinal encoded
            modulation mode

        Returns
        -------
        Tensor that stores target predicted class confidence (for the cross
        entropy loss) given a batch's ordinal encoded modulation mode
        """
        shape2d = [len(modeordenc), self.modclf.num_classes]

        linear_idx =\
            np.ravel_multi_index([np.arange(len(modeordenc)),
                                  np.array(modeordenc.cpu())],
                                 shape2d)

        target = np.zeros(shape2d)
        target.ravel()[linear_idx] = 15

        return torch.Tensor(target).softmax(axis=1).to(self.device)
