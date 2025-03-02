"""
Panoradio HF machine learning dataset analysis functions
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import colorcet as cc
import cmasher as cm
import colormaps as cmaps
import numpy as np
import pandas as pd
import scipy.signal as ssig
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import stft


def compute_stft(data_row,
                 signal_data,
                 **kwargs):
    """
    Computes a Short-Time Fourier Transform of a Panoradio HF radio signal
    classification dataset signal.

    Parameters
    ----------
    data_row: int
        Radio signal classification dataset 2-D matrix row

    signal_data: numpy.ndarray
        (# of signals) x (# of I/Q samples) numpy array that stores I/Q data

    kwargs: dict (Optional)
        Dictionary that stores mapping of optional arguments and their
        corresponding value

    Returns
    -------
    t: float
        STFT temporal sampling vector

    f: float
        STFT frequency axis sampling vector

    zxx:
        2-D matrix that stores the STFT frequency magnitude
    """
    window = kwargs.get("window", "hamming")
    nperseg = kwargs.get("nperseg", 240)
    noverlap = kwargs.get("noverlap", 180)

    f, t, zxx = stft(signal_data[data_row, :],
                     fs=6e3,
                     window=window,
                     nperseg=nperseg,
                     noverlap=noverlap,
                     return_onesided=False)

    is_pos_f = f >= 0
    is_neg_f = f < 0

    f = np.concatenate([f[is_neg_f], f[is_pos_f]])

    zxx = np.concatenate([zxx[is_neg_f, :],
                          zxx[is_pos_f, :]])

    return t, f, np.abs(zxx)


def init_modeid_index(train_data_tags,
                      snr_to_plot=25):
    """
    Initializes a dictionary that stores the training data sample
    indices corresponding to each modeid for the requested SNR

    Parameters
    ----------
    train_data_tags: pandas.DataFrame
        Data frame that identifies the mode & SNR for each Panoradio HF
        training dataset record

    snr_to_plot: float (Optional)
        Signal to Noise Ratio (SNR) used to segment the STFT plot data

    Returns
    -------
    modeid_index: dict
        Dictionary that stores the training data sample indices
        corresponding to each modeid for the requested SNR
    """
    modeid_index = {}

    for modeid in train_data_tags["mode"].unique():

        select_row =\
            np.logical_and(train_data_tags["snr"] == snr_to_plot,
                           train_data_tags["mode"] == modeid)

        modeid_index[modeid] =\
            train_data_tags.loc[select_row,
                                "index"].values

    return modeid_index


def compute_modeid_nsamples(modeid_index):
    """
    Determines the number of samples for each modulation mode (at
    a specific SNR)

    Parameters
    -------
    modeid_index : dict
        Dictionary that stores the training data sample indices
        corresponding to each modeid for the requested SNR

    Returns
    -------
    modeid_nsamples : pandas.DataFrame
        Number of samples for each modulation mode (at a specific SNR)
    """
    modeid_nsamples =\
        {key: len(values) for key, values in modeid_index.items()}

    modeid_nsamples = pd.DataFrame(pd.Series(modeid_nsamples))

    modeid_nsamples.rename(columns={0: "count"},
                           inplace=True)

    modeid_nsamples.sort_values("count",
                                inplace=True)

    return modeid_nsamples.reset_index()


def init_plot_index(modeid_index,
                    **kwargs):
    """
    Initializes a dictionary that stores a training dataset index
    corresponding to each of the 18 modulation modes

    Parameters
    ----------
    modeid_index : dict
        Dictionary that stores the training data sample indices
        corresponding to each modeid for the requested SNR

    kwargs : dict
        Dictionary that stores optional parameters

        pick_random_sample : bool
            Set to True to plot random modeid samples

        fixed_sample_idx : int
            Fixed sample index that should be less than the
            minimum number of modeid training data samples

    Returns
    -------
    plt_index : dict
        Dictionary that stores a training dataset index corresponding
        to each of the 18 modulation modes
    """
    pick_random_sample =\
        kwargs.get("pick_random_sample", False)

    fixed_sample_idx =\
        kwargs.get("fixed_sample_idx", 0)

    plt_index = {}
    for modeid in modeid_index.keys():

        if pick_random_sample:

            sample_idx = np.random.choice(modeid_index[modeid],
                                          size=1,
                                          replace=False)[0]
        # -------------------------------------------------
        else:
            nmodeid_samples = len(modeid_index[modeid])

            if fixed_sample_idx >= nmodeid_samples:

                errmsg =\
                    f"fixed_sample_idx: {fixed_sample_idx} > "

                errmsg += f"# of {modeid} samples {nmodeid_samples}"

                raise IndexError(errmsg)
            # -----------------------------------------------
            else:
                sample_idx =\
                    modeid_index[modeid][fixed_sample_idx]

        plt_index[modeid] = sample_idx

    return plt_index


def init_modeid_inphase(modeid,
                        modeid_index,
                        train_data):
    """
    Initializes a numpy.ndarray that stores the in-phase samples of
    the requested modulation mode (for a given SNR)

    Parameters
    ----------
    modeid : str
        String that refers to a specific modulation mode

    modeid_index : dict
        Dictionary that stores the training data sample indices
        corresponding to each modeid for the requested SNR

    train_data: numpy.ndarray
        (# of signals) x (# of I/Q samples) numpy array that stores the
        training I/Q data

    Returns
    -------
    modeid_inphase : numpy.ndarray
        In-phase samples of the requested modulation mode (for a given SNR)
    """
    nsamples = len(modeid_index[modeid])

    modeid_inphase = np.zeros((nsamples,
                               train_data.shape[1]))

    for write_row, read_row in enumerate(modeid_index[modeid]):
        modeid_inphase[write_row, :] = np.real(train_data[read_row, :])

    return modeid_inphase


def compute_modeid_stft(plt_index,
                        train_data):
    """
    Computes the Short-Time Fourier Transform (STFT) for each of the 18
    Panoradio HF dataset modulation modes

    Parameters
    ----------
    plt_index : dict
        Dictionary that stores a training dataset index corresponding
        to each of the 18 modulation modes

    train_data: numpy.ndarray
        (# of signals) x (# of I/Q samples) numpy array that stores the
        training I/Q data

    Returns
    -------
    tms: numpy.ndarray
        STFT temporal sampling [ms]

    fhz: numpy.ndarray
        STFT frequency axis sampling [Hz]

    zxx: dict
        Stores the estimated STFT for each of the 18 Panoradio HF dataset
        modulation modes
    """
    zxx = dict()
    vmin = sys.float_info.max
    vmax = sys.float_info.min

    for modeid in plt_index.keys():

        t, fhz, zxx[modeid] =\
            compute_stft(plt_index[modeid],
                         train_data)

        cur_min = zxx[modeid].min()
        cur_max = zxx[modeid].max()

        if cur_min < vmin:
            vmin = cur_min

        if cur_max > vmax:
            vmax = cur_max

    del cur_min, cur_max
    vmax_norm = 1.0 / vmax

    for modeid in plt_index.keys():
        zxx[modeid] *= vmax_norm
        zxx[modeid] = 10 * np.log(zxx[modeid])

    return t * 1E3, fhz, zxx


def plot_stft(tms,
              fhz,
              zxx):
    """
    Plots the Short-Time Fourier Transform (STFT) for each of the 18
    Panoradio HF dataset modulation modes

    Parameters
    ----------
    tms: numpy.ndarray
        STFT temporal sampling [ms]

    fhz: numpy.ndarray
        STFT frequency axis sampling [Hz]

    zxx: dict
        Stores the estimated STFT for each of the 18 Panoradio HF dataset
        modulation modes

    Returns
    -------
    None
    """
    mpl.style.use("seaborn-v0_8-paper")

    h_f, h_ax = plt.subplots(3, 6,
                             figsize=(9, 6),
                             sharex=True,
                             sharey=True,
                             layout="compressed")
    subplot_idx = 0
    modeid_list = list(zxx.keys())

    cmap = cc.linear_protanopic_deuteranopic_kbjyw_5_95_c25
    cmap = mpl.colors.ListedColormap(cmap)

    for row in range(h_ax.shape[0]):
        for col in range(h_ax.shape[1]):
            modeid = modeid_list[subplot_idx]

            h_img =\
                h_ax[row, col].pcolormesh(tms,
                                          fhz,
                                          zxx[modeid],
                                          vmin=-70,
                                          vmax=0,
                                          cmap=cmap)

            if row == (h_ax.shape[0] - 1):

                h_ax[row, col].set_xlabel("Time [ms]",
                                          fontweight="bold")

                for tick in h_ax[row, col].xaxis.get_majorticklabels():
                    tick.set_fontweight('bold')

            if col == 0:

                h_ax[row, col].set_ylabel("Freqeuncy [Hz]",
                                          fontweight="bold")

                for tick in h_ax[row, col].yaxis.get_majorticklabels():
                    tick.set_fontweight('bold')

            h_ax[row, col].set_title(modeid,
                                     fontweight="bold")
            subplot_idx += 1

    cbar = h_f.colorbar(h_img,
                        ax=h_ax.ravel().tolist(),
                        shrink=0.95,
                        label=r"$\bf{dB Magnitude}$")

    for tick in cbar.ax.yaxis.get_major_ticks():
        tick.label2.set_fontweight('bold')

    plt.savefig("stft.png",
                bbox_inches="tight")


def plot_modeid_inphase(modeid_index,
                        train_data,
                        **kwargs):
    """
    Plots the inphase samples for a set of signals

    Parameters
    ----------
    modeid_index : dict
        Dictionary that stores the training data sample indices
        corresponding to each modeid for the requested SNR

    train_data: numpy.ndarray
        (# of signals) x (# of I/Q samples) numpy array that stores the
        training I/Q data

    Returns
    -------
    None
    """
    vmin = sys.float_info.max
    vmax = sys.float_info.min
    modeid_inphase = {}

    for cur_modeid in modeid_index.keys():

        modeid_inphase[cur_modeid] =\
            init_modeid_inphase(cur_modeid,
                                modeid_index,
                                train_data)

        modeid_vmin = modeid_inphase[cur_modeid].min()
        modeid_vmax = modeid_inphase[cur_modeid].max()

        if modeid_vmin < vmin:
            vmin = modeid_vmin

        if modeid_vmax > vmax:
            vmax = modeid_vmax

        modeid_list = list(modeid_inphase.keys())

    mpl.style.use("seaborn-v0_8-paper")

    h_f, h_ax = plt.subplots(3, 6,
                             figsize=(9, 6),
                             sharex=True,
                             sharey=True,
                             layout="compressed")

    delta_t_ms = 1e3 / 6e3
    nsamples_to_plot = kwargs.get("nsamples_to_plot", 10)
    sample_idx = np.arange(nsamples_to_plot) + 1
    font_size = 10
    mode_idx = 0

    for row in range(h_ax.shape[0]):
        for col in range(h_ax.shape[1]):

            cur_modeid = modeid_list[mode_idx]

            cur_modeid_inphase =\
                modeid_inphase[cur_modeid][:nsamples_to_plot, :]

            tms = np.arange(cur_modeid_inphase.shape[1]) * delta_t_ms

            h_img =\
                h_ax[row, col].pcolormesh(tms,
                                          sample_idx,
                                          cur_modeid_inphase,
                                          vmin=vmin,
                                          vmax=vmax,
                                          cmap=cm.viola)

            h_ax[row, col].invert_yaxis()
            h_ax[row, col].set_title(cur_modeid,
                                     fontsize=font_size,
                                     fontweight="bold")

            h_ax[row, col].set_xlabel("Time [ms]",
                                      fontsize=font_size,
                                      fontweight="bold")

            for tick in h_ax[row, col].xaxis.get_majorticklabels():
                tick.set_fontweight('bold')
                tick.set_fontsize(font_size)

            if col == 0:

                h_ax[row, col].set_ylabel("Sample #",
                                          fontsize=font_size,
                                          fontweight="bold")

            for tick in h_ax[row, col].yaxis.get_majorticklabels():
                tick.set_fontweight('bold')
                tick.set_fontsize(font_size)

            mode_idx = mode_idx + 1

    cbar = h_f.colorbar(h_img,
                        ax=h_ax.ravel().tolist(),
                        shrink=0.95,
                        label=r"$\bf{In-phase}$")

    for tick in cbar.ax.yaxis.get_major_ticks():
        tick.label2.set_fontweight('bold')
        tick.label2.set_fontsize(font_size)

    plt.savefig("inphase.png",
                bbox_inches="tight")


def plot_fft_spectrum(batch_data):
    """
    Plots a heatmap that illustrates the dB FFT magnitude spectrum for a
    batch

    Parameters
    ----------
    batch_data: numpy.ndarray
        [batch size x 2 x # of samples] numpy nd-array that stores
        the (I / Q) signal samples corresponding to a batch

    Returns
    -------
    None
    """
    mpl.style.use("seaborn-v0_8-paper")

    cbatch_data = np.array(batch_data)
    cbatch_data = cbatch_data[:, 0, :] + 1j * cbatch_data[:, 1, :]
    batch_data_fft = np.fft.fftshift(np.fft.fft(cbatch_data, axis=1))
    batch_data_fft = 10 * np.log(np.abs(batch_data_fft))

    norm_freqs = np.arange(batch_data_fft.shape[1])
    dc_pt = int(batch_data_fft.shape[1] / 2)

    norm_freqs[dc_pt:] =\
        norm_freqs[dc_pt:] - batch_data_fft.shape[1]

    norm_freqs = np.fft.fftshift(norm_freqs / len(norm_freqs))

    dbmag_stats =\
        pd.Series(batch_data_fft.ravel()).describe().to_dict()

    h_fig, h_ax = plt.subplots(1, 1, figsize=(4, 3))

    h_img =\
        h_ax.pcolormesh(norm_freqs,
                        np.arange(batch_data_fft.shape[0]) + 1,
                        batch_data_fft,
                        cmap=cm.voltage,
                        vmin=dbmag_stats["50%"],
                        vmax=dbmag_stats["max"])

    h_fig.colorbar(h_img,
                   ax=h_ax,
                   shrink=0.95,
                   label=r"$\bf{dB Magnitude}$")

    h_ax.invert_yaxis()


def compute_db_spectrum(signal_samples):

    db_spectrum = {}

    db_spectrum["value"] =\
        fft(signal_samples *
            ssig.windows.hamming(len(signal_samples)))

    db_spectrum["value"] = np.abs(fftshift(db_spectrum["value"]))
    db_spectrum["value"] = 20 * np.log10(db_spectrum["value"])

    db_spectrum["normfreqs"] =\
        fftshift(fftfreq(len(db_spectrum["value"])))

    return db_spectrum


def compare_modulations(bpsk_signal,
                        qpsk_signal):

    bpsk_db_spectrum = compute_db_spectrum(bpsk_signal)
    qpsk_db_spectrum = compute_db_spectrum(qpsk_signal)

    bpsk_color = cmaps.discrete_Bg.colors[0]
    qpsk_color = cmaps.discrete_Bg.colors[-1]

    h_fig, h_ax = plt.subplots(2, 2, figsize=(6, 4))

    h_ax[0, 0].plot(bpsk_db_spectrum["normfreqs"],
                    bpsk_db_spectrum["value"],
                    color=bpsk_color)

    h_ax[0, 1].scatter(np.real(bpsk_signal),
                       np.imag(bpsk_signal),
                       alpha=0.1,
                       color=bpsk_color)

    h_ax[1, 0].plot(qpsk_db_spectrum["normfreqs"],
                    qpsk_db_spectrum["value"],
                    color=qpsk_color)

    h_ax[1, 1].scatter(np.real(qpsk_signal),
                       np.imag(qpsk_signal),
                       alpha=0.1,
                       color=qpsk_color)

    for row in range(h_ax.shape[0]):

        h_ax[row, 0].set_ylim([-10, 50])
        h_ax[row, 1].set_xlim([-2, 2])
        h_ax[row, 1].set_ylim([-2, 2])

        h_ax[row, 0].set_xlabel("Normalized Frequency")
        h_ax[row, 0].set_ylabel("dB Magnitude")

        h_ax[row, 1].set_xlabel("In-phase")
        h_ax[row, 1].set_ylabel("Quadrature")

        for col in range(h_ax.shape[1]):
            h_ax[row, col].grid(True)

    for col in range(h_ax.shape[1]):
        h_ax[0, col].set_title("BPSK")
        h_ax[1, col].set_title("QPSK")

    h_fig.tight_layout()
