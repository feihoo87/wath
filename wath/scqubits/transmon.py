import itertools
from collections import defaultdict
from functools import reduce

import numpy as np
from numpy.linalg import eigh, eigvalsh
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from scipy.optimize import minimize

CAP_UNIT = 1e-15  # fF
FREQ_UNIT = 1e9  # GHz
RESISTANCE_UNIT = 1.0  # Ohm


def connected_components(nodes, junctions):
    graph = defaultdict(set)
    for a, b in junctions:
        graph[a].add(b)
        graph[b].add(a)

    visited = set()

    comps = []

    for n in nodes:
        if n in visited:
            continue

        stack = [n]
        comp = []
        visited.add(n)

        while stack:
            x = stack.pop()
            comp.append(x)
            for y in graph[x]:
                if y not in visited:
                    visited.add(y)
                    stack.append(y)

        comps.append(comp)

    return comps


def complete_incidence_matrix(nodes, junctions):
    n = len(nodes)
    idx = {name: i for i, name in enumerate(nodes)}

    comps = connected_components(nodes, junctions)

    # 必须满足可逆条件
    assert len(junctions) + len(comps) == n, \
        "junctions must form a spanning forest"

    S = np.zeros((n, n), dtype=int)

    # ----- junction rows -----
    for i, (a, b) in enumerate(junctions):
        S[i, idx[a]] = 1
        S[i, idx[b]] = -1

    # ----- component rows -----
    offset = len(junctions)
    for k, comp in enumerate(comps):
        for node in comp:
            S[offset + k, idx[node]] = 1

    return S


def tenser(matrices):
    return reduce(np.kron, matrices)


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


def Ec_matrix_from_capacitance(C, nodes, junctions):
    """
    C: capacitance matrix in fF
    nodes: list of nodes
    junctions: list of junctions
    return: Ec matrix in GHz
    """
    from scipy.constants import e, h

    # unit: GHz
    Ec = np.linalg.inv(C * CAP_UNIT) * 2 * e**2 / h / FREQ_UNIT
    S = complete_incidence_matrix(nodes, junctions)
    V = np.linalg.inv(S)
    Ec = V.T @ Ec @ V
    return Ec[:len(junctions), :len(junctions)]


def eig_singal_qubit(Ec, EJ, ng=0.0, levels=5, eps=1e-6):
    E = None

    for N in range(levels + 2, 100):
        n = np.arange(-N, N + 1) - ng
        w, v = eigh_tridiagonal(4 * Ec * n**2, -EJ * np.full(2 * N, 0.5))
        v = v[:, :levels]
        w = w[:levels]
        if E is not None:
            if np.all(np.abs(E - w) < eps):
                break
        E = w
    return w, v.T.conj() @ np.diag(n) @ v, v, N


def spectrum(Ec=100.0,
             EJS=6000.0,
             flux=0.0,
             ng=0.0,
             d=0.0,
             levels=5,
             decouple=False,
             return_psi=False):
    """
    Ec: capacitance matrix in GHz
    EJS: symmetric Josephson energy in GHz
    flux: flux in Phi_0
    ng: charge bias
    d: asymmetry parameter
    levels: number of levels

    return: spectrum in GHz
    """
    Ec = np.atleast_2d(Ec)
    EJS, flux = np.atleast_1d(EJS, flux)

    num_qubits = Ec.shape[0]
    w0, n_ops = [], []
    for Ec_, EJ in zip(np.diag(Ec), flux_to_EJ(flux, EJS, d)):
        e, n, v, N = eig_singal_qubit(Ec_, EJ, ng=ng, levels=levels)
        w0.append(e[:levels])
        v = v[:, :levels]
        n_ops.append(n)

    w0 = np.sum(np.meshgrid(*w0, copy=False, indexing='ij'),
                axis=0).reshape(-1)

    if decouple or num_qubits == 1:
        w0 = np.sort(w0)
        return w0[1:] - w0[0]

    I = np.eye(levels)

    H0 = np.diag(w0)
    Hc = np.zeros_like(H0)
    for i, j in itertools.combinations(range(num_qubits), r=2):
        Hc += 8 * Ec[i, j] * tenser(
            [n_ops[k] if k == i or k == j else I for k in range(num_qubits)])

    if return_psi:
        w, psi = eigh(H0 + Hc)
        return w[1:] - w[0], psi[:, 1:], Hc
    else:
        w = eigvalsh(H0 + Hc)
        return w[1:] - w[0]
