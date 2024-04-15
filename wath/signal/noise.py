from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import welch


def noise(psd: Callable[[ArrayLike], ArrayLike],
          tlist: ArrayLike | None = None,
          freq: ArrayLike | None = None,
          phase: ArrayLike | None = None) -> ArrayLike:
    """
    Generate noise signal with given spectrum.

    Parameters
    ----------
    psd : callable
        Function that returns the power spectral density of the noise signal.
    tlist : array_like, optional
        Time list. If not given, it will be generated from `freq`.
    freq : array_like, optional
        Frequency list. If not given, it will be generated from `tlist`.
    phase : array_like, optional
        Phase of the noise signal. If not given, it will be generated randomly.
    
    Returns
    -------
    array_like
        The generated noise signal.
    """
    if freq is None and tlist is None:
        raise ValueError("Either `tlist` or `freq` must be given.")
    if freq is None:
        freq = np.fft.fftfreq(len(tlist), d=tlist[1] - tlist[0])
    df = freq[1] - freq[0]
    if phase is None:
        phase = np.random.random(len(freq)) * np.pi * 2
    spec = np.sqrt(psd(freq) * df) * np.exp(1j * phase) * len(freq)
    return np.fft.ifft(spec).real


def psd(sig: ArrayLike, sample_rate=1):
    """
    Calculate the power spectral density of a signal.

    Parameters
    ----------
    sig : array_like
        The signal.
    sample_rate : float, optional
        Sample rate of the signal.

    Returns
    -------
    array_like
        The power spectral density.
    """
    spec = np.abs(np.fft.fft(sig) / len(sig))**2 / sample_rate
    freq = np.fft.fftfreq(len(sig), d=1 / sample_rate)
    return freq, spec


def Wiener_process(tlist: ArrayLike) -> ArrayLike:
    """
    Generate Wiener process.

    Parameters
    ----------
    tlist : array_like
        Time list.

    Returns
    -------
    array_like
        The generated Wiener process.
    """
    return np.cumsum(
        np.random.normal(0, np.sqrt(tlist[1] - tlist[0]), len(tlist)))
