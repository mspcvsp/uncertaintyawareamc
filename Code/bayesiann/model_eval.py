"""
Deep learning model evaluation tools
"""
import joblib as jl
import numpy as np
import pandas as pd
import lightning as pl
from sklearn.metrics import classification_report, confusion_matrix
from .ccnn import ClassicalCNN
from .deepcnn import DeepCNN
from .resnet import ResidualNetwork
from .allconvnet import AllConvNet
from .data import get_data_dir, IQDataModel
from .net_utils import get_model_checkpoint_dir, parse_snrid
from .net_utils import get_ordenc_intid_lut


def evaluate_model_performance(modelid_checkpoint_map,
                               **kwargs):
    """
    Evaluates the performance of a set of trained deep learning models

    Parameters
    ----------
    modelid_checkpoint_map: dict
        Map of network architecture identifiers to the corresponding model
        checkpoint files

    kwargs: dict
        Optional parameters
    """
    trainer = pl.Trainer()
    datamodule = IQDataModel(**kwargs)
    snrid_acc = []

    for modelid in modelid_checkpoint_map.keys():

        model_checkpoint_pth = get_model_checkpoint_dir(modelid)
        checkpoint_file = modelid_checkpoint_map[modelid]

        model_checkpoint_pth =\
            model_checkpoint_pth.joinpath(checkpoint_file)

        if modelid == "classical-cnn":
            model = ClassicalCNN.load_from_checkpoint(model_checkpoint_pth)
        # -----------------------------------------
        elif modelid == "all-conv-net":
            model = AllConvNet.load_from_checkpoint(model_checkpoint_pth)
        # -----------------------------------------
        elif modelid == "deep-cnn":
            model = DeepCNN.load_from_checkpoint(model_checkpoint_pth)
        # -----------------------------------------
        elif modelid == "resnet":
            model = ResidualNetwork.load_from_checkpoint(model_checkpoint_pth)

        predictions = trainer.predict(model,
                                      datamodule=datamodule)

        snrid_clf_report, _ =\
            evaluate_model_predictions(predictions)

        model_snrid_acc = init_snrid_accuracy(snrid_clf_report)
        model_snrid_acc["modelid"] = modelid
        snrid_acc.append(model_snrid_acc)

    return pd.concat(snrid_acc)


def init_snrid_accuracy(snrid_clf_report):
    """
    Initializes a Pandas DataFrame that stores a network's modulation
    classification as a function of Signal-to-Noise Ratio

    Parameters
    ----------
    snrid_clf_report: dict
        Classification report for each dataset SNR

    Returns
    -------
    Pandas DataFrame that stores a network's modulation classification as
    a function of Signal-to-Noise Ratio
    """
    snrid_acc =\
        pd.Series({key: snrid_clf_report[key]["accuracy"]
                   for key in snrid_clf_report})

    snrid_acc = pd.DataFrame(snrid_acc)
    snrid_acc.reset_index(inplace=True)

    snrid_acc.rename(columns={0: "accuracy",
                              "index": "snrid"}, inplace=True)

    snrid_acc["snr"] =\
        snrid_acc["snrid"].apply(lambda elem: parse_snrid(elem))

    return snrid_acc


class ModelPredictionFormatter(object):
    """
    Class that formats model predictions
    """

    def __init__(self,
                 **kwargs):
        """
        ModelPredictionFormatter class constructor

        Parameters
        ----------
        self: ModelPredictionFormatter
            ModelPredictionFormatter class object reference

        kwargs: dict
            Optional parameters
        """
        data_dir = get_data_dir(**kwargs)

        self.mode_ordenc =\
            jl.load(data_dir.joinpath("modeordenc.jl"))

        self.snr_ordenc =\
            jl.load(data_dir.joinpath("snrordenc.jl"))

        self.modeint_modeid_lut =\
            get_ordenc_intid_lut(self.mode_ordenc)

        self.snrint_snrid_lut =\
            get_ordenc_intid_lut(self.snr_ordenc)

    def __call__(self,
                 batch_preds):
        """
        Formats model predictions

        Parameters
        ----------
        self: ModelPredictionFormatter
            ModelPredictionFormatter class object reference

        batch_preds: list
            Tuple that stores the estimated modulation model confidence,
            ordinal encoded true modulation mode & ordinal encoded SNR

        Returns
        -------
        batch_preds: pandas.DataFrame
            Pandas DataFrame that stores formated batch predictions
        """
        mode_conf_preds =\
            pd.DataFrame(np.array(batch_preds[0]))

        mode_conf_preds.rename(columns=self.modeint_modeid_lut,
                               inplace=True)

        predictedmodeid =\
            np.zeros(mode_conf_preds.shape[0],
                     dtype=object)

        for index, row in mode_conf_preds.iterrows():

            predictedmodeid[index] =\
                self.modeint_modeid_lut[row.argmax()]

        mode_ints =\
            np.array(batch_preds[1]).reshape(-1, 1)

        mode_conf_preds["truemodeid"] =\
            self.mode_ordenc.inverse_transform(mode_ints).flatten()

        mode_conf_preds["predictedmodeid"] = predictedmodeid

        snr_ints =\
            np.array(batch_preds[2]).reshape(-1, 1)

        mode_conf_preds["snrid"] =\
            self.snr_ordenc.inverse_transform(snr_ints).flatten()

        return mode_conf_preds


def evaluate_model_predictions(predictions):
    """
    Generates a classification report and confusion matrix for a set of
    modulation mode predicitions as a function of SNR

    Parameters
    ----------
    predictions: list
        List of tuples that stores the predicted modulation mode confidence,
        ordinal encoded modulation mode, and ordinal encoded SNR for each
        datasplit batch

    Returns
    -------
    snrid_clf_report: dict
        Classification report for each dataset SNR

    confusion_matrix: dict
        Dictionary of Pandas DataFrames that stores the confusion matrix for
        each dataset SNR
    """
    pred_fmt = ModelPredictionFormatter()

    mode_conf_preds = \
        pd.concat([pred_fmt(elem) for elem in predictions])

    snrid_clf_report = dict()
    snrid_conf_mat = dict()
    modeids = list(mode_conf_preds.columns[:18])

    for snrid in mode_conf_preds["snrid"].value_counts().keys():

        select_row = mode_conf_preds["snrid"] == snrid

        truemodeid =\
            mode_conf_preds.loc[select_row, "truemodeid"].values

        predictedmodeid =\
            mode_conf_preds.loc[select_row, "predictedmodeid"].values

        snrid_clf_report[snrid] =\
            classification_report(truemodeid,
                                  predictedmodeid,
                                  zero_division=0,
                                  output_dict=True)

        cur_conf_mat =\
            confusion_matrix(truemodeid,
                             predictedmodeid,
                             normalize="true")

        snrid_conf_mat[snrid] =\
            pd.DataFrame(cur_conf_mat,
                         index=modeids,
                         columns=modeids)

    return snrid_clf_report, snrid_conf_mat
