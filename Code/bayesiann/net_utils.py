"""
Deep learning algorithm utility functions
"""
import os
from pathlib import Path
import torch.nn as nn
import torch.nn.init as init
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


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
    path_elems = ["/home",
                  os.getenv("USER"),
                  "panoradio_hf",
                  kwargs.get("model_checkpoint_dir", "ModelCheckpoints"),
                  net_arch_id]

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
    min_delta = kwargs.get("min_delta", 0.01)

    mlflow_tracking_uri = kwargs.get("mlflow_tracking_uri",
                                     get_mlflow_tracking_uri())

    checkpoint_callback = ModelCheckpoint(
        dirpath=get_model_checkpoint_dir(net_arch_id,
                                         **kwargs),
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}')

    mlf_logger =\
        MLFlowLogger(experiment_name=net_arch_id,
                     tracking_uri=f"file:{mlflow_tracking_uri}")

    early_stop_callback =\
        EarlyStopping(monitor="val_acc",
                      min_delta=min_delta,
                      patience=patience,
                      verbose=False,
                      mode="max")

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
