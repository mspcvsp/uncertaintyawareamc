"""
Deep learning algorithm utility functions
"""
import inspect
import os
from pathlib import Path
import torch.nn as nn
import torch.nn.init as init
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger 
import pytorch_lightning as pl


def init_weights(m):
    """
    Linear layer initialiation function

    Parameters
    ----------
    m: torch.nn.modules object

    Returns
    -------
    None
    """
    if isinstance(m, nn.Linear):

        init.kaiming_normal_(m.weight,
                             mode='fan_in',
                             nonlinearity='relu')

        m.bias.data.fill_(0.01)
    # --------------------------------
    elif isinstance(m, nn.Conv1d):

        init.kaiming_normal_(m.weight,
                             mode='fan_in',
                             nonlinearity='relu')

        if m.bias is not None:
            m.bias.data.fill_(0.01)


def get_model_checkpoint_dir(net_arch_id,
                             **kwargs):
    """
    Returns the full path to the top-level model checkpoint directory

    Parameters
    ----------
    net_arch_id: str
        Neural network architecture identifier

    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Top-level model checkpoint directory name

    Returns
    -------
    data_dir: str
        Full path to the top-level model checkpoint directory name
    """
    path_elems = list(Path(inspect.getfile(inspect.currentframe())).parts)[:-3]

    path_elems.extend([kwargs.get("model_checkpoint_dir", "ModelCheckpoints"),
                       net_arch_id])

    return Path(*path_elems)


def get_mlflow_tracking_uri():
    """
    Returns an MLFlow tracking URI

    Parameters
    ----------
    None

    Returns
    -------
    mlflow_tracking_uri: str
        MLFlow tracking URI
    """
    mlflow_tracking_uri =\
        Path(*["/home",
               os.getenv("USER"),
               "ml-runs"])

    return mlflow_tracking_uri


def initialize_pl_trainer(net_arch_id,
                          **kwargs):
    """
    Initializes a Pytorch Trainer class object constructor inputs

    Parameters
    ----------
    net_arch_id: str
        String that refers to a deep learning algorithm architecture

    kwargs: dict
        Optional parameters

    Returns
    -------
    callbacks: list
        List of Pytorch Lighting Trainer callback(s)

    mlf_logger: MLFlowLogger
        MLFlow logger class object
    """
    patience = kwargs.get("patience", 5)
    min_delta = kwargs.get("min_delta", 0.001)

    mlflow_tracking_uri = kwargs.get("mlflow_tracking_uri",
                                     get_mlflow_tracking_uri())

    checkpoint_callback =\
        ModelCheckpoint(
            save_top_k=2,
            monitor="validation_loss",
            dirpath=get_model_checkpoint_dir(net_arch_id,
                                             **kwargs),
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}')
    
    early_stop_callback =\
        EarlyStopping(monitor="validation_loss",
                      min_delta=min_delta,
                      patience=patience,
                      verbose=False,
                      mode="min")
    
    mlf_logger =\
        MLFlowLogger(experiment_name=net_arch_id,
                     tracking_uri=f"file:{mlflow_tracking_uri}")
    
    callbacks = [checkpoint_callback,
                 early_stop_callback]

    return callbacks, mlf_logger


def get_ordenc_intid_lut(ordenc):
    """
    Returns a dictionary that stores a mapping between ordinal encoded
    classes and the corresponding class names

    Parameters
    ----------
    ordenc: OrdinalEncoder
        OrdinalEncoder class object

    Returns
    -------
    ordenc_intid_lut: dict
        Dictionary that stores a mapping between ordinal encoded
        classes and the corresponding class names
    """
    values = ordenc.categories_[0]
    enc_values = ordenc.transform(values.reshape(-1, 1))

    return dict(zip(enc_values.flatten().astype(int), values))


def parse_snrid(snrid):
    """
    Parses a string encoded Signal-to-Noise (SNR) level

    Parameters
    ----------
    snrid: str
        String encoded Signal-to-Noise (SNR) level

    Returns
    -------
    snr: float
        Signal-to-Noise (SNR) level
    """
    snr = snrid.replace("snr", "")
    return float(snr.replace("minus", "-"))


def compute_conv1d_lout(l_in,
                        kernel_size,
                        **kwargs):
    """
    Computes the number of 1-D convolution output samples

    Parameters
    ----------
    l_in: int
        Number of 1-D convolution input samples

    kerne_size: int
        1-D convolution kernel size

    kwargs: dict
        Optional parameters
    """
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)

    l_out = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
    l_out = l_out / stride + 1
    return int(l_out)


class ModelCheckpoint(object):

    def __init__(self,
                 net_arch_id,
                 checkpoint_file):

        checkpoint_pth =\
            get_model_checkpoint_dir(net_arch_id).joinpath(checkpoint_file)

        self.model_checkpoint = torch.load(checkpoint_pth)

    def get_subnet_state_dict(self,
                              subnet_id):

        state_dict = OrderedDict()

        for key in self.model_checkpoint["state_dict"].keys():

            if subnet_id in key:

                state_dict[key.replace(subnet_id + ".", "")] =\
                    self.model_checkpoint["state_dict"][key]

        return state_dict
