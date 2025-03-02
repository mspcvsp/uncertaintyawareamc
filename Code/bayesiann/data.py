"""
Panoradio HF machine learning dataset functions
"""
import joblib as jl
import os
import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
import lightning as pl
import polars
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def load_data_tags(**kwargs) -> pd.DataFrame:
    """
    Loads data tags used to split Panoradio HF data into training,
    validation, & test datasets

    Parameters
    ----------
    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Panoradio HF data set directory name

    Returns
    -------
    data_tags: pandas.DataFrame
        Data frame that identifies the mode & SNR for each Panoradio HF
        dataset record
    """
    csv_file_path =\
        get_data_dir(**kwargs).joinpath("dataset_panoradio_hf_tags.csv")

    data_tags = polars.read_csv(csv_file_path)

    data_tags =\
        data_tags.select(
            polars.col("idx"),
            polars.col(" mode").alias("mode"),
            polars.col(" snr").alias("snr").str.strip_chars()
        ).with_columns(
            polars.concat_str([
                polars.col("mode"),
                polars.col("snr")
            ], separator="",
            ).alias("modesnr")
        ).select(
            polars.col("idx"),
            polars.col("mode"),
            polars.col("snr").cast(polars.Float64),
            polars.col("modesnr").str.replace_all('-', 'minus', literal=True)
        )

    return data_tags.to_pandas()


def initialize_data_split_idx(data_tags,
                              **kwargs):
    """
    Initializes training, validation, & test data split indexing.

    Parameters
    ----------
    data_tags: pandas.DataFrame
        Data frame that identifies the mode & SNR for each Panoradio HF
        dataset record

    kwargs: dict
        Dictionary that stores optional parameters

        train_size : float
            Training data set fraction of the complete dataset and
            validation data set fraction of the resulting training data

        random_state0: int
            Training / test data train_test_split random state

        random_state1: int
            Training / validation train_test_split random state

    Returns
    -------
    data_split_idx: dict
        Dictionary that stores training, validation, & test data split
        sample numbers
    """
    train_size = kwargs.get("train_size", 0.8)
    random_state0 = kwargs.get("random_state0", 588194253)
    random_state1 = kwargs.get("random_state1", 400839924)

    data_split_idx = {}

    data_split_idx["train"], data_split_idx["test"] =\
        train_test_split(data_tags["idx"].values,
                         train_size=train_size,
                         random_state=random_state0)

    data_split_idx["train"], data_split_idx["validation"] =\
        train_test_split(data_split_idx["train"],
                         train_size=train_size,
                         random_state=random_state1)

    return data_split_idx


def get_data_dir(**kwargs):
    """
    Returns the full path to the Panoradio HF dataset directory

    Parameters
    ----------
    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Panoradio HF data set directory name

    Returns
    -------
    data_dir: str
        Full path to the Panoradio HF dataset directory
    """
    path_elems = ["/home",
                  os.getenv("USER"),
                  kwargs.get("data_dir", "Data"),
                  "panoradio_hf"]

    return Path(*path_elems)


def get_datasplit_data_path(datasplit,
                            **kwargs):
    """
    Returns the full path to a datasplit data *.npy file

    Parameters
    ----------
    datasplit: str
        String that refers to a specific data split (e.g. "train")

    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Panoradio HF data set directory name

    Returns
    -------
    datasplit_data_path: PosixPath
        full path to a datasplit data *.npy file
    """
    data_dir = get_data_dir(**kwargs)

    return data_dir.joinpath(f"{datasplit}_data.npy")


def get_datasplit_data_tags_path(datasplit,
                                 **kwargs):
    """
    Returns the full path to a datasplit data tags *.csv file

    Parameters
    ----------
    datasplit: str
        String that refers to a specific data split (e.g. "train")

    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Panoradio HF data set directory name

    Returns
    -------
    datasplit_data_tags_path: PosixPath
        Full path to a datasplit data tags *.csv file
    """
    data_dir = get_data_dir(**kwargs)

    return data_dir.joinpath(f"{datasplit}_data_tags.csv")


def initialize_data_splits(**kwargs):
    """
    Initializes data split data *.npy and data tags *.csv files

    Parameters
    ----------
    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Panoradio HF data set directory name

    Returns
    -------
    None
    """
    if get_datasplit_data_path("train",
                               **kwargs).exists() is False:

        data_tags = load_data_tags(**kwargs)

        data_split_idx = initialize_data_split_idx(data_tags)

        init_datasplit_data_tags(data_tags,
                                 data_split_idx,
                                 **kwargs)

        data_dir = get_data_dir(**kwargs)
        iq_data = np.load(data_dir.joinpath("dataset_hf_radio.npy"))

        for cur_split in data_split_idx.keys():

            cur_idx = data_split_idx[cur_split]
            datasplit_data = iq_data[cur_idx, :].copy()

            datasplit_data =\
                np.concatenate([datasplit_data.real[:, np.newaxis, :],
                                datasplit_data.imag[:, np.newaxis, :]],
                               axis=1)

            np.save(get_datasplit_data_path(cur_split),
                    datasplit_data)


def init_datasplit_data_tags(data_tags,
                             data_split_idx,
                             **kwargs):
    """
    Initializes a data tags *.csv file for the training, validation & test
    data splits

    Parameters
    ----------
    data_tags: pandas.DataFrame
        Data frame that identifies the mode & SNR for each Panoradio HF
        dataset record

    data_split_idx: dict
        Dictionary that stores training, validation, & test data split
        sample numbers

    kwargs: dict
        Dictionary that stores optional parameters

        data_dir : str
            Panoradio HF data set directory name

    Returns
    -------
    None
    """
    for cur_split in data_split_idx.keys():

        cur_idx = data_split_idx[cur_split]

        datasplit_data_tags =\
            data_tags.iloc[cur_idx, :].copy()

        datasplit_data_tags["snrid"] =\
            datasplit_data_tags.apply(lambda row: init_snrid(row),
                                      axis=1)

        datasplit_data_tags.reset_index(inplace=True,
                                        drop=True)

        cur_split_modeid =\
            datasplit_data_tags["mode"].values.copy().reshape(-1, 1)

        cur_split_snrid =\
            datasplit_data_tags["snrid"].values.copy().reshape(-1, 1)

        if cur_split == "train":

            mode_ordenc = OrdinalEncoder()
            snr_ordenc = OrdinalEncoder()

            datasplit_data_tags["modeordenc"] =\
                mode_ordenc.fit_transform(cur_split_modeid)

            datasplit_data_tags["snrordenc"] =\
                snr_ordenc.fit_transform(cur_split_snrid)
        # --------------------------------------------
        else:

            datasplit_data_tags["modeordenc"] =\
                mode_ordenc.transform(cur_split_modeid)

            datasplit_data_tags["snrordenc"] =\
                snr_ordenc.transform(cur_split_snrid)

        datasplit_data_tags.to_csv(get_datasplit_data_tags_path(cur_split,
                                                                **kwargs),
                                   index=False)

    data_dir = get_data_dir(**kwargs)

    jl.dump(mode_ordenc, data_dir.joinpath("modeordenc.jl"))
    jl.dump(snr_ordenc, data_dir.joinpath("snrordenc.jl"))


def init_snrid(row):
    """
    Returns a string that refers to a specific Signal-to-Noise (SNR) level

    Parameters
    ----------
    row: pandas.Series
        Panoradio HF machine learning dataset tags pandas.DataFrame row

    Returns
    -------
    snrid: str
        String that refers to a specific Signal-to-Noise (SNR) level
    """
    return "snr" + row.modesnr.replace(row["mode"], "")


def init_modeid_mode_lut():
    """
    Initializes a mode identifier / mode Look-Up Table (LUT)

    Parameters
    ----------
    None

    Returns
    -------
    modeid_mode_lut: dict
        Mode identifier / mode Look-Up Table (LUT)
    """
    modeid_mode_lut =\
        {'am': "AM broadcast",
         'dominoex11': "DominoEx",
         'fax': "radiofax",
         'morse': "Morse Code",
         'mt63_1000': "MT63",
         'navtex': "Navtex / Sitor-B",
         'olivia16_1000': "Olivia 16/1000",
         'olivia16_500': "Olivia 16/500",
         'olivia32_1000': "Olivia 32/1000",
         'olivia8_250': "Olivia 8/250",
         'psk31': "PSK31",
         'psk63': "PSK63",
         'qpsk31': "QPSK31",
         'rtty100_850': "RTTY 100/850",
         'rtty45_170': "RTTY 45/170",
         'rtty50_170': "RTTY 50/170",
         'lsb': "SSB (Lower)",
         'usb': "SSB (Lower)"}

    return modeid_mode_lut


def init_modeid_waveform_lut():
    """
    Initializes a mode identifier / waveform Look-Up Table (LUT)

    Parameters
    ----------
    None

    Returns
    -------
    modeid_mode_lut: dict
        Mode identifier / waveform Look-Up Table (LUT)
    """
    modeid_waveform_lut =\
        {'am': "AM",
         'dominoex11': "18-MFSK",
         'fax': "radiofax",
         'morse': "OOK",
         'mt63_1000': "Multi-carrier",
         'navtex': "FSK 170 Hz shift",
         'olivia16_1000': "16-MFSK",
         'olivia16_500': "16-MFSK",
         'olivia32_1000': "32-MFSK",
         'olivia8_250': "8-MFSK",
         'psk31': "PSK",
         'psk63': "PSK",
         'qpsk31': "QPSK",
         'rtty100_850': "FSK",
         'rtty45_170': "FSK",
         'rtty50_170': "FSK",
         'lsb': "LSB",
         'usb': "USB"}

    return modeid_waveform_lut


def iqdata_to_complex(iqdata):
    """
    Converts interleaved in-phase & quadrature data to complex valued
    numpy ndarray

    Parameters
    ----------
    iqdata: numpy.ndarray
        Real-valued array that stores interleaved I/Q data

    Returns
    ------
    complex_data: numpy.ndarray
        Complex-valued array that stores I/Q data
    """
    ndims = len(iqdata.shape)

    if ndims == 2:
        complex_data = iqdata[0, :] + 1j * iqdata[1, :]
    elif ndims == 3:
        complex_data = iqdata[:, 0, :] + 1j * iqdata[:, 1, :]

    return complex_data


class IQDataTransformer(object):
    """
    Inphase / Quadrature (I/Q) data augmenter
    """
    def __init__(self,
                 **kwargs):
        """
        Initializes an IQDataTransformer class object

        Parameters
        ----------
        self : IQDataTransformer
            IQDataTransformer class object handle

        kwargs : dict
            Optional parameters

            rng_seed: int
                Random number generator seed

        Returns
        -------
        self : IQDataTransformer
            IQDataTransformer class object handle
        """
        rng_seed = kwargs.get("rng_seed", 495508588)
        self.rng = Generator(MT19937(SeedSequence(rng_seed)))

    def __call__(self,
                 batch_iqdata_in):
        """
        Adds a random phase offset to each row of an I/Q data batch

        Complex multiplication
        ----------------------
        (a + jb) * (c + jd) = a * c + j(a * d) + j(b * c) + j ** 2 * (a * d)
        = (ac - bd) + j(ad + bc)

        Parameters
        ----------
        self : IQDataTransformer
            IQDataTransformer class object handle

        batch_iqdata_in : numpy.ndarray
            [Batch size] x 2048 2-D array that stores a batch of (I/Q) data

        Returns
        -------
        batch_iqdata_out : numpy.ndarray
            [Batch size] x 2048 2-D array that stores a batch of (I/Q) data
        """
        ndims = len(batch_iqdata_in.shape)

        if ndims == 2:

            rnd_phase =\
                np.exp(2j * np.pi * self.rng.uniform(size=1))[0]

            batch_iqdata_out = batch_iqdata_in.copy()

            # (ac - bd) + j(ad + bc)
            batch_iqdata_out[0, :] =\
                (batch_iqdata_out[0, :] * rnd_phase.real -
                 batch_iqdata_out[1, :] * rnd_phase.imag)

            batch_iqdata_out[1, :] =\
                (batch_iqdata_out[0, :] * rnd_phase.imag +
                 batch_iqdata_out[1, :] * rnd_phase.real)
        # ---------------------------------------------
        elif ndims == 3:

            batch_size = batch_iqdata_in.shape[0]
            rnd_phase = np.exp(2j * np.pi * self.rng.uniform(size=batch_size))
            batch_iqdata_out = np.zeros((batch_size, 2, 2048))

            # (ac - bd) + j(ad + bc)
            batch_iqdata_out[:, 0, :] =\
                (batch_iqdata_in[:, 0, :].T * rnd_phase.real -
                 batch_iqdata_in[:, 1, :].T * rnd_phase.imag).T

            batch_iqdata_out[:, 1, :] =\
                (batch_iqdata_in[:, 0, :].T * rnd_phase.imag +
                 batch_iqdata_in[:, 1, :].T * rnd_phase.real).T

        return batch_iqdata_out


class IQDataset(Dataset):
    """
    Inphase / Quadrature (I/Q) dataset class
    """

    def __init__(self,
                 datasplit,
                 **kwargs):
        """
        Inphase / Quadrature (I/Q) dataset class constructor

        Parameters
        ----------
        self: IQDataset
            Class reference

        datasplit : str
            Datasplit indentifier (e.g. "train", "validation", or "test")

        kwargs: dict
            Dictionary that stores optional parameters

            data_dir : str
                Panoradio HF data set directory name

            transform: object
                i/Q data augmentation class object

        Returns
        -------
        self: IQDataset
            Class reference
        """
        self.data_tags =\
            pd.read_csv(get_datasplit_data_tags_path(datasplit,
                                                     **kwargs))

        self.data =\
            np.load(get_datasplit_data_path(datasplit),
                    mmap_mode="r")

        self.transform = kwargs.get("transform", None)

    def __len__(self):
        """
        Returns the (I/Q dataset length)

        Parameters
        ----------
        self: IQDataset
            Class reference

        Returns
        -------
        len: int
            Dataset size
        """
        return self.data.shape[0]

    def __getitem__(self,
                    idx):
        """
        Returns one or more I/Q dataset elements

        Parameters
        ----------
        self: IQDataset
            Class reference

        idx: iterable
            Dataset indices

        Returns
        -------
        batch_data: torch.Tensor
            I/Q dataset batch

        modeordenc: int
            Ordinal modulation mode encoding

        snrordenc: int
            Signal-to-Noise (SNR) ratio ordinal encoding
        """
        modeordenc =\
            self.data_tags.iloc[idx]["modeordenc"].astype(int)

        snrordenc =\
            self.data_tags.iloc[idx]["snrordenc"].astype(int)

        batch_data = self.data[idx, :].copy()

        if self.transform is not None:
            batch_data = self.transform(batch_data)

        return torch.from_numpy(batch_data).float(), modeordenc, snrordenc


class IQDataModel(pl.LightningDataModule):
    """
    I/Q data Pytorch Lighting data module
    """

    def __init__(self,
                 **kwargs):
        """
        I/Q data module class constructor

        Parameters
        ----------
        self: IQDataModel class reference

        kwargs: dict
            Optional parameters

        Returns
        ----------
        self: IQDataModel class reference
        """
        super().__init__()
        self.data_dir = kwargs.get("data_dir", "Data")
        self.batch_size = kwargs.get("batch_size", 1024)
        self.num_workers = kwargs.get("num_workers", 15)

        self.predict_datasplit = kwargs.get("predict_datasplit",
                                            "validation")

    def setup(self, stage: str):
        """
        Method that initializes training, validation & test datasets

        Parameters
        ----------
        self: IQDataModel class reference

        stage: str
            String that identifies the current module development
            lifecycle stage that can be one of the following strings:
            - fit
            - validate
            - test
            - predict

        Returns
        ----------
        self: IQDataModel class reference
        """
        self.train = IQDataset("train",
                               data_dir=self.data_dir,
                               transform=IQDataTransformer())

        self.validation = IQDataset("validation",
                                    data_dir=self.data_dir)

        self.test = IQDataset("test",
                              data_dir=self.data_dir)

        if self.predict_datasplit == "validation":

            self.predict = IQDataset("validation",
                                     data_dir=self.data_dir)
        # ---------------------------------------------------
        elif self.predict_datasplit == "test":

            self.predict = IQDataset("test",
                                     data_dir=self.data_dir)
        # ---------------------------------------------------
        else:
            raise ValueError("Invalid predict datasplit: " +
                             f"{self.predict_datasplit}")

    def train_dataloader(self):
        """
        Returns a training dataset Pytorch Dataloader

        Parameters
        ----------
        self: IQDataModel class reference

        Returns
        -------
        train_dl: DataLoader
            DataLoader class
        """
        return DataLoader(self.train,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Returns a validation dataset Pytorch Dataloader

        Parameters
        ----------
        self: IQDataModel class reference

        Returns
        -------
        train_dl: DataLoader
            DataLoader class
        """
        return DataLoader(self.validation,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns a test dataset Pytorch Dataloader

        Parameters
        ----------
        self: IQDataModel class reference

        Returns
        -------
        train_dl: DataLoader
            DataLoader class
        """
        return DataLoader(self.test,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          batch_size=self.batch_size)

    def predict_dataloader(self):
        """
        Returns a predict dataset Pytorch Dataloader

        Parameters
        ----------
        self: IQDataModel class reference

        Returns
        -------
        train_dl: DataLoader
            DataLoader class
        """
        return DataLoader(self.predict,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          batch_size=self.batch_size)

    def taredown(self, stage: str):
        """
        Class "destructor" method

        Parameters
        ----------
        self: IQDataModel class reference

        stage: str
            String that identifies the current module development
            lifecycle stage that can be one of the following strings:
            - fit
            - validate
            - test
            - predict

        Returns
        ----------
        None
        """
        pass
