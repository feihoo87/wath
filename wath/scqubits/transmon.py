import itertools
from functools import reduce

import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from scipy.optimize import minimize

CAP_UNIT = 1e-15  # fF
FREQ_UNIT = 1e9  # GHz
RESISTANCE_UNIT = 1.0  # Ohm
"""
常见超导材料及其对应的超导能隙：

| 超导材料 | 超导能隙 (Δ, $\\mu$eV)|
|---------|--------------------|
| 铝 (Al) |         170        |
| 钽 (Ta) |         700-1,000  |
| 铌 (Nb) |         1500       |
| 汞 (Hg) |         16         |
| 铍 (Be) |         300-450    |
| 镍 (Ni) |         200        |
| 锡 (Sn) |         1100       |
| 铅 (Pb) |         1400       |
| 镧钡铜氧化物 (LBCO) |  20,000-30,000   |
| 高温钡镧铜氧化物 (BSCCO) | 10,000-15,000 |

请注意，这些值可能因材料的纯度、制备条件等因素而有所变化。
"""


class Transmon():

    def __init__(self, **kw):
        self.Ec = 0.2
        self.EJ = 20

        self.d = 0
        if kw:
            self._set_params(**kw)

    def _set_params(self, **kw):
        if {"EJ", "Ec", "d"} <= set(kw):
            return self._set_params_EJS_Ec_d(kw['EJ'], kw['Ec'], kw['d'])
        elif {"EJ", "Ec"} <= set(kw):
            return self._set_params_EJ_Ec(kw['EJ'], kw['Ec'])
        elif {"f01", "alpha"} <= set(kw):
            if 'ng' not in kw:
                return self._set_params_f01_alpha(kw['f01'], kw['alpha'])
            else:
                return self._set_params_f01_alpha(kw['f01'], kw['alpha'],
                                                  kw['ng'])
        elif {"f01_max", "f01_min"} <= set(kw):
            if {"alpha1", "alpha2"} <= set(kw):
                return self._set_params_f01_max_min_alpha(
                    kw['f01_max'], kw['f01_min'], kw['alpha1'], kw['alpha2'],
                    kw.get('ng', 0))
            elif {"alpha"} <= set(kw):
                return self._set_params_f01_max_min_alpha(
                    kw['f01_max'], kw['f01_min'], kw['alpha'], kw['alpha'],
                    kw.get('ng', 0))
            elif {"alpha1"} <= set(kw):
                return self._set_params_f01_max_min_alpha(
                    kw['f01_max'], kw['f01_min'], kw['alpha1'], kw['alpha1'],
                    kw.get('ng', 0))
        raise TypeError('_set_params() got an unexpected keyword arguments')

    def _set_params_EJ_Ec(self, EJ, Ec):
        self.Ec = Ec
        self.EJ = EJ

    def _set_params_EJS_Ec_d(self, EJS, Ec, d):
        self.Ec = Ec
        self.EJ = EJS
        self.d = d

    def _set_params_f01_alpha(self, f01, alpha, ng=0):
        Ec = -alpha
        EJ = (f01 - alpha)**2 / 8 / Ec

        def err(x, target=(f01, alpha)):
            EJ, Ec = x
            levels = self._levels(Ec, EJ, ng=ng)
            f01 = levels[1] - levels[0]
            f12 = levels[2] - levels[1]
            alpha = f12 - f01
            return (target[0] - f01)**2 + (target[1] - alpha)**2

        ret = minimize(err, x0=[EJ, Ec])
        self._set_params_EJ_Ec(*ret.x)

    def _set_params_f01_max_min_alpha(self,
                                      f01_max,
                                      f01_min,
                                      alpha1,
                                      alpha2=None,
                                      ng=0):
        if alpha2 is None:
            alpha2 = alpha1

        Ec = -alpha1
        EJS = (f01_max - alpha1)**2 / 8 / Ec
        d = (f01_min + Ec)**2 / (8 * EJS * Ec)

        def err(x, target=(f01_max, alpha1, f01_min, alpha2)):
            EJS, Ec, d = x
            levels = self._levels(Ec, self._flux_to_EJ(0, EJS, d), ng=ng)
            f01_max = levels[1] - levels[0]
            f12 = levels[2] - levels[1]
            alpha1 = f12 - f01_max

            levels = self._levels(Ec, self._flux_to_EJ(0.5, EJS, d), ng=ng)
            f01_min = levels[1] - levels[0]
            f12 = levels[2] - levels[1]
            alpha2 = f12 - f01_min

            return (target[0] - f01_max)**2 + (target[1] - alpha1)**2 + (
                target[2] - f01_min)**2 + (target[3] - alpha2)**2

        ret = minimize(err, x0=[EJS, Ec, d])
        self._set_params_EJS_Ec_d(*ret.x)

    @staticmethod
    def _flux_to_EJ(flux, EJS, d=0):
        F = np.pi * flux
        EJ = EJS * np.sqrt(np.cos(F)**2 + d**2 * np.sin(F)**2)
        return EJ

    @staticmethod
    def _levels(Ec, EJ, ng=0.0, grid_size=None, select_range=(0, 10)):
        x_arr = np.asarray(EJ)
        is_scalar = x_arr.ndim == 0

        if grid_size is None:
            grid_size = max(select_range) + 11
        n = np.arange(grid_size) - grid_size // 2
        n_levels = select_range[1] - select_range[0]
        results = np.zeros((len(x_arr), n_levels))
        diag = 4 * Ec * (n - ng)**2

        for i, ej_val in enumerate(EJ.reshape(-1)):
            off_diag = -ej_val / 2 * np.ones(grid_size - 1)
            w = eigvalsh_tridiagonal(diag,
                                     off_diag,
                                     select='i',
                                     select_range=select_range)
            results[i] = w[1:] - w[0]
        if is_scalar:
            return results[0]
        else:
            return results.reshape(*EJ.shape, n_levels)

    def levels(self, flux=0, ng=0, select_range=(0, 10), grid_size=None):
        return self._levels(self.Ec,
                            self._flux_to_EJ(flux, self.EJ, self.d),
                            ng,
                            select_range=select_range,
                            grid_size=grid_size)

    @property
    def EJ1_EJ2(self):
        return (1 + self.d) / (1 - self.d)

    def chargeParityDiff(self, flux=0, ng=0, k=0):
        a = self.levels(flux, ng=0 + ng)
        b = self.levels(flux, ng=0.5 + ng)

        return (a[..., 1 + k] - a[..., k]) - (b[..., 1 + k] - b[..., k])


def transmon_levels(x,
                    period,
                    offset,
                    EJS,
                    Ec,
                    d,
                    ng=0.0,
                    grid_size=None,
                    select_range=(0, 10)):
    q = Transmon(EJ=EJS, Ec=Ec, d=d)
    return q.levels(flux=(x - offset) / period,
                    ng=ng,
                    select_range=select_range,
                    grid_size=grid_size)


def mass(C):
    """
    C: capacitance matrix in fF

    return: mass matrix in GHz^-1
    """
    from scipy.constants import e, h

    a = np.diag(C)
    b = C - np.diag(a)
    c = np.sum(b, axis=0)
    C = np.diag(a + c) - b

    # convert unit of capacitance, make sure energy unit is GHz
    M = C * CAP_UNIT / (4 * e**2 / h / FREQ_UNIT)
    return M


def Rn_to_EJ(Rn, gap=200, T=10):
    """
    Rn: normal resistance in Ohm
    gap: superconducting gap in ueV
    T: temperature in mK

    return: EJ in GHz
    """
    from scipy.constants import e, h, hbar, k, pi

    Delta = gap * e * 1e-6
    Ic = pi * Delta / (2 * e * Rn * RESISTANCE_UNIT) * np.tanh(
        Delta / (2 * k * T * 1e-3))
    EJ = Ic * hbar / (2 * e)
    return EJ / h / FREQ_UNIT


def flux_to_EJ(flux, EJS, d=0):
    """
    flux: flux in Phi_0
    EJS: symmetric Josephson energy in GHz
    d: asymmetry parameter
        EJ1 / EJ2 = (1 + d) / (1 - d)
    """
    F = np.pi * flux
    EJ = EJS * np.sqrt(np.cos(F)**2 + d**2 * np.sin(F)**2)
    return EJ


def n_op(N=5):
    return np.diag(np.arange(-N, N + 1))


# def cos_phi_op(N=5):
#     from scipy.sparse import diags
#     return diags(
#         [np.full((2 * N, ), 0.5),
#          np.full(
#              (2 * N, ), 0.5), [0.5], [0.5]], [1, -1, 2 * N, -2 * N]).toarray()


def cos_phi_op(N=5):
    from scipy.sparse import diags
    return diags([np.full(
        (2 * N, ), 0.5), np.full((2 * N, ), 0.5)], [1, -1]).toarray()


# def sin_phi_op(N=5):
#     from scipy.sparse import diags
#     return diags(
#         [np.full((2 * N, ), 0.5j),
#          np.full((2 * N, ), -0.5j), [-0.5j], [0.5j]],
#         [1, -1, 2 * N, -2 * N]).toarray()


def sin_phi_op(N=5):
    from scipy.sparse import diags
    return diags([np.full(
        (2 * N, ), 0.5j), np.full((2 * N, ), -0.5j)], [1, -1]).toarray()


def phi_op(N=5):
    from scipy.fft import fft, ifft, ifftshift

    k = ifftshift(np.arange(-N, N + 1) * np.pi / N)
    psi = np.eye(k.shape[0])
    T = fft(psi, overwrite_x=True)
    T *= k
    return ifft(T, overwrite_x=True)


def H_C(C, N=5, ng=None, decouple=False):
    num_qubits = C.shape[0]
    if ng is None:
        ng = np.zeros(num_qubits)
    elif isinstance(ng, (int, float)):
        ng = np.full(num_qubits, ng)
    elif isinstance(ng, (list, tuple)):
        ng = np.array(ng)
    else:
        raise ValueError("ng must be a number or a list of numbers")

    A = np.linalg.inv(mass(C))

    n = n_op(N)
    I = np.eye(n.shape[0])

    if decouple:
        return [0.5 * A[i, i] * (n - ng[i])**2 for i in range(num_qubits)]

    n_ops = []
    for i in range(num_qubits):
        n_ops.append(
            reduce(np.kron,
                   [n - ng[i] if j == i else I for j in range(num_qubits)]))

    ret = np.zeros_like(n_ops[0], dtype=float)

    for i, n_i in enumerate(n_ops):
        for j, n_j in enumerate(n_ops):
            ret += n_i * A[i, j] / 2 * n_j
    return ret


def H_phi(Rn, flux, d=0, gap=200, T=10, N=5, decouple=False):
    num_qubits = Rn.shape[0]
    EJ = flux_to_EJ(flux, Rn_to_EJ(Rn, gap, T), d)
    op = cos_phi_op(N)
    I = np.eye(op.shape[0])

    if decouple:
        return [-EJ[i] * op for i in range(num_qubits)]

    ret = np.zeros((op.shape[0]**num_qubits, op.shape[0]**num_qubits),
                   dtype=float)
    for i in range(num_qubits):
        ret -= EJ[i] * reduce(np.kron,
                              [op if j == i else I for j in range(num_qubits)])

    return ret


def spectrum(C=100.0,
             Rn=6000.0,
             flux=0.0,
             ng=0.0,
             d=0.0,
             levels=5,
             gap=200.0,
             T=10.0,
             selector=None,
             decouple=False):
    """
    C: capacitance matrix in fF
    Rn: normal resistance in Ohm
    flux: flux in Phi_0
    ng: charge bias
    d: asymmetry parameter
    levels: number of levels
    gap: superconducting gap in ueV
    T: temperature in mK

    return: spectrum in GHz
    """
    if decouple:
        w0 = [
            np.linalg.eigvalsh(hc + hp) for hc, hp in zip(
                H_C(C, N=levels, ng=ng, decouple=True),
                H_phi(Rn, flux, d=d, gap=gap, T=T, N=levels, decouple=True))
        ]
        w0 = np.sort([np.sum(l) for l in itertools.product(*w0)])
        return w0[1:] - w0[0]
    H = H_C(C, N=levels, ng=ng) + H_phi(Rn, flux, d=d, gap=gap, T=T, N=levels)
    if selector is None:
        w = np.linalg.eigvalsh(H)
        return w[1:] - w[0]

    assert len(
        selector) == C.shape[0], "selector must be a list of qubit indices"

    w, psi = np.linalg.eigh(H)

    H0 = [
        hc + hp for hc, hp in zip(
            H_C(C, N=levels, ng=ng, decouple=True),
            H_phi(Rn, flux, d=d, gap=gap, T=T, N=levels, decouple=True))
    ]
    w0, baises = [], []
    for h in H0:
        e, v = np.linalg.eigh(h)
        w0.append(e)
        baises.append(v)
    psi0 = reduce(np.kron, [baises[q][:, i] for q, i in enumerate(selector)])
    overlap = np.abs(psi0.T @ psi)**2

    return overlap[:, 1:], w[1:] - w[0]
