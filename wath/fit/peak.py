import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import peak_widths

from ..signal.func import peaks


class Normalize:
    """
    A class which, when called, linearly normalizes data into the
    ``[0.0, 1.0]`` interval.
    """

    def __init__(self, vmin=None, vmax=None):
        """
        Parameters
        ----------
        vmin, vmax : float or None
            If *vmin* and/or *vmax* is not given, they are initialized from the
            minimum and maximum value, respectively, of the first input
            processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.

        Notes
        -----
        Returns 0 if ``vmin == vmax``.
        """
        self._vmin = vmin
        self._vmax = vmax

    @property
    def vmin(self):
        return self._vmin

    @property
    def vmax(self):
        return self._vmax

    @staticmethod
    def process_value(value):
        """
        Homogenize the input *value* for easy and efficient normalization.

        *value* can be a scalar or sequence.

        Returns
        -------
        result : masked array
            Masked array with the same shape as *value*.
        is_scalar : bool
            Whether *value* is a scalar.

        Notes
        -----
        Float dtypes are preserved; integer types with two bytes or smaller are
        converted to np.float32, and larger types are converted to np.float64.
        Preserving float32 when possible, and using in-place operations,
        greatly improves speed for large arrays.
        """
        is_scalar = not np.iterable(value)
        if is_scalar:
            value = [value]
        dtype = np.min_scalar_type(value)
        if np.issubdtype(dtype, np.integer) or dtype.type is np.bool_:
            # bool_/int8/int16 -> float32; int32/int64 -> float64
            dtype = np.promote_types(dtype, np.float32)
        # ensure data passed in as an ndarray subclass are interpreted as
        # an ndarray. See issue #6622.
        mask = np.ma.getmask(value)
        data = np.asarray(value)
        result = np.ma.array(data, mask=mask, dtype=dtype, copy=True)
        return result, is_scalar

    def __call__(self, value):
        """
        Normalize *value* data in the ``[vmin, vmax]`` interval into the
        ``[0.0, 1.0]`` interval and return it.

        Parameters
        ----------
        value
            Data to normalize.

        Notes
        -----
        If not already initialized, ``self.vmin`` and ``self.vmax`` are
        initialized using ``self.autoscale_None(value)``.
        """
        result, is_scalar = self.process_value(value)

        if self.vmin is None or self.vmax is None:
            self.autoscale_None(result)
        # Convert at least to float, without losing precision.
        (vmin, ), _ = self.process_value(self.vmin)
        (vmax, ), _ = self.process_value(self.vmax)
        if vmin == vmax:
            result.fill(0)  # Or should it be all masked?  Or 0.5?
        elif vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        else:
            # ma division is very slow; we can take a shortcut
            resdat = result.data
            resdat -= vmin
            resdat /= (vmax - vmin)
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until both vmin and vmax are set")
        (vmin, ), _ = self.process_value(self.vmin)
        (vmax, ), _ = self.process_value(self.vmax)

        if np.iterable(value):
            val = np.ma.asarray(value)
            return vmin + val * (vmax - vmin)
        else:
            return vmin + value * (vmax - vmin)

    def autoscale_None(self, A):
        """If vmin or vmax are not set, use the min/max of *A* to set them."""
        A = np.asanyarray(A)
        if self._vmin is None and A.size:
            self._vmin = A.min()
        if self._vmax is None and A.size:
            self._vmax = A.max()

    def scaled(self):
        """Return whether vmin and vmax are set."""
        return self.vmin is not None and self.vmax is not None


def peaks_fun(x, *args):
    n = (len(args) - 1) // 4
    p = []
    for i in range(n):
        center, sigma, gamma, amp = [*args[4 * i:4 * i + 4]]
        p.append((center, sigma, gamma, amp))
    bg = args[-1]
    return peaks(x, p, bg)


def _fit_single_peak(x, y):
    i = np.argmax(y)
    widths, *_ = peak_widths(
        y,
        [i],
        rel_height=0.5,
    )
    width = max(widths[0], 3)
    width = min(i, len(x) - i - 1, width)

    gamma = width / len(x) * (x[-1] - x[0])

    f0 = x[i]
    offset = np.median(y)
    amp = y[i] - offset

    start = max(0, i - 4 * int(width))
    stop = min(len(x), i + 4 * int(width))

    popt, pcov = curve_fit(
        peaks_fun,
        x[start:stop],
        y[start:stop], [f0, 0, gamma, amp, offset],
        bounds=([x[0], 0, (x[1] - x[0]) / 2, 0,
                 np.min(y) - np.max(y)], [
                     x[-1], x[-1] - x[0], x[-1] - x[0],
                     np.max(y) - np.min(y),
                     np.max(y)
                 ]),
        method='trf')
    return popt


def fit_peaks(x, y, n=1):
    """
    Fit peaks in y(x) with n peaks.

    Args:
        x: np.array
        y: np.array
        n: number of peaks to fit

    Returns:
        p: list of (center, width, amp, shape)
        bg: background
        See also: waveforms.math.signal.func.peaks
    """
    norm_x = Normalize(vmax=np.max(x), vmin=np.min(x))
    norm_y = Normalize(vmax=np.max(y), vmin=np.min(y))

    ydata = norm_y(y)
    xdata = norm_x(x)

    p = []
    for i in range(n):
        popt = _fit_single_peak(xdata, ydata)
        ydata -= peaks_fun(xdata, *popt)
        f0, sigma, gamma, amp, offset = popt
        p.extend([f0, sigma, gamma, amp])

    ydata = norm_y(y)
    p.append(np.median(ydata))
    popt, pcov = curve_fit(peaks_fun,
                           xdata,
                           ydata,
                           p0=p,
                           method='trf',
                           maxfev=10000)
    p = []
    for i in range(n):
        center, sigma, gamma, amp = [*popt[4 * i:4 * i + 4]]
        p.append(
            (norm_x.inverse(center), sigma, gamma * (np.max(x) - np.min(x)),
             amp * (np.max(y) - np.min(y))))
    bg = norm_y.inverse(popt[-1])
    return p, bg
