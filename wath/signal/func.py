from math import lcm

import numpy as np
import scipy.constants as const
from scipy.signal import convolve, correlate, correlation_lags, resample
from scipy.special import voigt_profile


def Svv(f, T, Z=lambda f: 50 * np.ones_like(f)):
    """
    power spectral density of the series noise voltage

    Johnson-Nyquist noise
    https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    
    f : list of frequency
    T : temperature
    Z : frequency-dependent complex electrical impedance
        (default 50 Ohm)
    """
    from scipy.constants import h, k

    if callable(Z):
        R = np.real(Z(f))
    else:
        R = np.real(Z)
    x = h * f / (k * T)
    x, R, f, T = np.broadcast_arrays(x, R, f, T)
    ret = np.zeros_like(x)
    mask1 = x < 37
    mask2 = x >= 37
    ret[mask1] = 4 * k * T[mask1] * R[mask1] * x[mask1] / (np.exp(x[mask1]) -
                                                           1)
    ret[mask2] = 4 * h * f[mask2] * R[mask2] * np.exp(-x[mask2])
    return ret


def Sii(f, T, Z=lambda f: 50 * np.ones_like(f)):
    """
    power spectral density of the series noise voltage

    Johnson-Nyquist noise
    https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    
    f : list of frequency
    T : temperature
    Z : frequency-dependent complex electrical impedance
        (default 50 Ohm)
    """
    from scipy.constants import h, k

    if callable(Z):
        Y = np.real(1 / Z(f))
    else:
        Y = np.real(1 / Z)
    x = h * f / (k * T)
    x, Y, f, T = np.broadcast_arrays(x, Y, f, T)
    ret = np.zeros_like(x)
    mask1 = x < 37
    mask2 = x >= 37
    ret[mask1] = 4 * k * T[mask1] * Y[mask1] * x[mask1] / (np.exp(x[mask1]) -
                                                           1)
    ret[mask2] = 4 * h * f[mask2] * Y[mask2] * np.exp(-x[mask2])
    return ret


def atts(f, atts=[], input=None):
    """
    power spectral density at MXC
    
    f : list of frequency
    atts: list of tuples (temperature, attenuator)    
    """
    if input is not None:
        spec = input
    else:
        spec = 0.5 * Svv(f, 300)
    for T, att in atts:
        A = 10**(-att / 10)
        spec = spec / A + 0.5 * Svv(f, T) * (A - 1) / A
    return spec


def atts_and_heat(f, atts=[], input=None):
    """
    power spectral density at MXC
    
    f : list of frequency
    atts: list of tuples (temperature, attenuator)    
    """
    heat = np.zeros_like(f)
    if input is not None:
        spec = input
    else:
        spec = 0.5 * Svv(f, 300)
    for T, att in atts:
        A = 10**(-att / 10)
        heat += 300 / T * (A - 1) / A * spec
        spec = spec / A + 0.5 * Svv(f, T) * (A - 1) / A
    return spec, heat


def thermal_excitation(T, f01, *levels):
    """
    p1 / p0 = exp(-beta*E01)
    p0 + p1 = 1
    p1 = tanh(-beta * E01 / 2) / 2 + 1 / 2
    """
    if len(levels) == 0:
        return 0.5 * np.tanh(-0.5 * const.h * f01 / const.k / T) + 0.5
    else:
        levels = np.hstack([[f01], levels])
        pp = np.exp(-const.h * levels / const.k / T)
        p0 = 1 / (np.sum(pp) + 1)
        return pp * p0


def effective_temperature(p1, f01):
    """
    effective temperature of qubit
    """
    return -0.5 * const.h * f01 / const.k / np.arctanh(2 * p1 - 1)


def Z_in(w, ZL, l, Z0=50, v=1e8):
    """Impedance of the transmission line
    """
    a = 1j * np.tan(w * l / v)
    #return Z0*(np.tanh(1j*w*l/v)+np.arctanh(ZL/Z0))
    return Z0 * (ZL + Z0 * a) / (Z0 + ZL * a)


def S21(w, l, ZL, Z0=50, v=1e8):
    """S21 of the transmission line
    """
    z = Z_in(w, ZL, l, Z0, v) / Z0
    phi = w * l / v
    #return (1+Z0/z)*np.exp(-1j*w*l/v)/2+(1-Z0/z)*np.exp(1j*w*l/v)/2
    return np.cos(phi) - 1j / z * np.sin(phi)


def lorentzianAmp(x, x0, gamma):
    """lorentzian peak"""
    return 1 / (1 + ((x - x0) / gamma)**2)


def lorentzian(x, x0, gamma):
    """complex lorentzian peak
    
    lorentzianAmp(x, x0, gamma) = lorentzian(x, x0, gamma) * conj(lorentzian(x, x0, gamma))
    """
    return 1 / (1 + 1j * (x - x0) / gamma)


def gaussian(x, x0, sigma):
    """gaussian peak"""
    return np.exp(-0.5 * ((x - x0) / sigma)**2)


def lorentzianGaussian(x, x0, gamma, sigma):
    """complex lorentzian peak
    """
    if gamma == 0:
        return gaussian(x, x0, sigma / np.sqrt(2))
    elif sigma == 0:
        return lorentzian(x, x0, gamma)
    else:
        ker = gaussian(x - x.mean(), 0, sigma / np.sqrt(2))
        ker /= ker.sum()
        return convolve(lorentzian(x, x0, gamma),
                        ker,
                        mode='same',
                        method='fft')


def Voigt(x, x0, sigma, gamma):
    """Voigt profile

    The Voigt profile (named after Woldemar Voigt) is a probability distribution
    given by a convolution of a Cauchy-Lorentz distribution and a Gaussian
    distribution. It is often used in analyzing data from spectroscopy or diffraction.

    See also
    https://en.wikipedia.org/wiki/Voigt_profile
    """
    if sigma == 0 and gamma == 0:
        return np.where(x == x0, 1.0, 0)
    return voigt_profile(x - x0, sigma, gamma) / voigt_profile(0, sigma, gamma)
    from scipy.special import wofz
    if gamma == 0:
        return gaussian(x, x0, sigma) / sigma / np.sqrt(2 * np.pi)
    elif sigma == 0:
        return lorentzian(x, x0, gamma) / gamma / np.pi
    else:
        z = (x - x0 + 1j * gamma) / (sigma * np.sqrt(2))
        return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def lorentzianSpace(f0, gamma, numOfPoints):
    """frequencies for lorentzian peak

    Args:
        f0 : center frequency
        gamma : width
        numOfPoints : number of points
    
    Returns:
        frequencies: np.array
    """
    theta = np.linspace(-np.pi, np.pi, numOfPoints + 2)[1:-1]
    return f0 - gamma * np.tan(theta)


def peaks(x, peaks, background=0):
    """
    peaks: list of (center, sigma, gamma, amp)
    background: a float or ndarray with the same shape of `x`
    """
    ret = np.zeros_like(x)
    for center, sigma, gamma, amp in peaks:
        ret += amp * Voigt(x, center, sigma, gamma)

    return ret + background


def complexPeaks(x, peaks, background=0):
    """
    peaks: list of (center, width, amp)
    background: a float, complex or ndarray with the same shape of `x`
    """
    ret = np.zeros_like(x, dtype=complex)
    for x0, gamma, A, *_ in peaks:
        ret += A * lorentzian(x, x0, gamma)
    return ret + background


def decay(t, tau):
    """
    exponential decay
    """
    a = -(1 / np.asarray(tau))**(np.arange(len(tau)) + 1)
    a = np.hstack([a[::-1], [0]])
    return np.exp(np.poly1d(a)(t))


def oscillation(t, spec=((1, 1), ), amplitude=1, offset=0):
    """
    oscillation
    """
    ret = np.zeros_like(t, dtype=complex)
    for A, f in spec:
        ret += A * np.exp(2j * np.pi * f * t)
    return amplitude * np.real(ret) + offset


def correlation(in1, in2, sample_rate=1, sample_rate2=None):
    """
    cross-correlation of two signals

    Args:
        in1, in2: input signals
        sample_rate: sample rate of in1
        sample_rate2: sample rate of in2, if None, use sample_rate

    Returns:
        x: time lags
        y: cross-correlation
    """
    sample_rate1 = round(sample_rate)
    if sample_rate2 is None:
        sample_rate2 = round(sample_rate)
    else:
        sample_rate2 = round(sample_rate2)
    sample_rate = lcm(sample_rate1, sample_rate2)

    in1_len = len(in1) * sample_rate // sample_rate1
    in2_len = len(in2) * sample_rate // sample_rate2
    if in1_len > len(in1):
        in1 = resample(in1, in1_len)
    if in2_len > len(in2):
        in2 = resample(in2, in2_len)

    x = correlation_lags(in1_len, in2_len, mode='full') / float(sample_rate)
    y = correlate(in1, in2, mode='full', method='fft') / np.sqrt(
        correlate(in1, in1, mode='valid')[0] *
        correlate(in2, in2, mode='valid')[0])

    return x, y

