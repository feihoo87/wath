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
    """使用深度优先搜索算法找出电路中的所有连通分量。

    该函数用于分析超导电路的拓扑结构，识别由结连接在一起的节点组。
    每个连通分量代表一个电学上相互连接的电路部分。

    参数:
        nodes: 节点列表，可以是任意可哈希类型（如整数、字符串）。
        junctions: 结列表，每个元素是一个元组 (a, b)，表示节点 a 和 b 之间的约瑟夫森结连接。

    返回:
        连通分量列表，每个元素是一个包含该分量中所有节点的列表。

    示例:
        >>> nodes = [1, 2, 3, 4]
        >>> junctions = [(1, 2), (2, 3)]  # 节点 4 是孤立的
        >>> connected_components(nodes, junctions)
        [[1, 2, 3], [4]]
    """
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
    """构建电路的完整关联矩阵。

    关联矩阵描述了超导电路的拓扑结构，包含约瑟夫森结连接和连通分量（孤岛）信息。
    该矩阵用于将电容矩阵转换到适当的坐标系以计算充电能量。

    参数:
        nodes: 节点标识符列表。
        junctions: 结列表，每个元素是一个元组 (a, b)，表示连接节点 a 和 b 的约瑟夫森结。

    返回:
        numpy.ndarray: 一个 n×n 的关联矩阵 S，其中：
            - 前 len(junctions) 行表示结连接（+1 和 -1）
            - 剩余行表示连通分量（孤岛），对应节点位置为 1

    异常:
        AssertionError: 如果结不能形成生成森林（必须满足：len(junctions) + len(components) == len(nodes)）。

    示例:
        >>> nodes = [1, 2, 3]
        >>> junctions = [(1, 2), (2, 3)]
        >>> S = complete_incidence_matrix(nodes, junctions)
        >>> S.shape
        (3, 3)
    """
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
    """计算多个矩阵的克罗内克积（张量积）。

    这是构建多量子比特算符的实用函数，通过单量子比特算符的张量积来构造。
    在多体量子系统中，用于组合不同子空间上的算符。

    参数:
        matrices: 待相乘的 numpy 数组列表。

    返回:
        numpy.ndarray: 所有输入矩阵的克罗内克积。

    示例:
        >>> import numpy as np
        >>> sx = np.array([[0, 1], [1, 0]])  # Pauli-X 矩阵
        >>> I = np.eye(2)  # 单位矩阵
        >>> tenser([sx, I]).shape  # 两量子比特算符
        (4, 4)
    """
    return reduce(np.kron, matrices)


class Transmon():
    """Transmon 量子比特模型，用于能谱计算。

    Transmon 是一种超导量子比特，通过大分流电容对电荷噪声不敏感。
    该类提供了在不同条件下计算 Transmon 量子比特能级和其他性质的方法。

    基本参数:
        Ec (float): 充电能量，单位 GHz。默认值为 0.2 GHz。
            Ec = e²/(2C)，其中 C 是总电容。
        EJ (float): 约瑟夫森能量，单位 GHz。默认值为 20 GHz。
            EJ = IcΦ₀/(2π)，表示结的隧穿能量。
        d (float): 非对称 SQUID 的非对称参数。默认值为 0（对称）。
            EJ1/EJ2 = (1 + d) / (1 - d)，其中 EJ1 和 EJ2 是两个结的能量。

    参数初始化方式:
        该类支持多种参数初始化方式：
        - 直接设置 EJ、Ec: Transmon(EJ=20, Ec=0.2)
        - 带非对称参数: Transmon(EJ=20, Ec=0.2, d=0.1)
        - 从 f01 和 alpha 反推: Transmon(f01=5.0, alpha=-0.3)
        - 从磁通可调参数: Transmon(f01_max=6.0, f01_min=4.0, alpha=-0.3)

    示例:
        >>> q = Transmon(EJ=20, Ec=0.2)
        >>> levels = q.levels()  # 获取能级
        >>> levels[:3]  # 前三个能级（相对于基态）
        array([0.        , 4.963..., 9.726...])
    """

    def __init__(self, **kw):
        self.Ec = 0.2
        self.EJ = 20

        self.d = 0
        if kw:
            self._set_params(**kw)

    def _set_params(self, **kw):
        """根据提供的关键字参数自动选择合适的参数设置方法。

        支持的参数组合:
            - {"EJ", "Ec", "d"}: 设置对称约瑟夫森能量、充电能量和非对称参数
            - {"EJ", "Ec"}: 直接设置约瑟夫森能量和充电能量
            - {"f01", "alpha"}: 从基态-第一激发态跃迁频率和非谐性反推参数
            - {"f01_max", "f01_min"}: 从磁通可调频率的最大/最小值反推参数

        参数:
            **kw: 关键字参数，根据提供的参数组合自动路由到相应的设置方法。

        异常:
            TypeError: 如果提供的参数组合不被支持。
        """
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
        """直接设置约瑟夫森能量和充电能量。

        参数:
            EJ (float): 约瑟夫森能量，单位 GHz。
            Ec (float): 充电能量，单位 GHz。
        """
        self.Ec = Ec
        self.EJ = EJ

    def _set_params_EJS_Ec_d(self, EJS, Ec, d):
        """设置对称约瑟夫森能量、充电能量和非对称参数。

        参数:
            EJS (float): 对称约瑟夫森能量（flux=0 时的有效 EJ），单位 GHz。
            Ec (float): 充电能量，单位 GHz。
            d (float): 非对称参数，EJ1/EJ2 = (1 + d) / (1 - d)。
        """
        self.Ec = Ec
        self.EJ = EJS
        self.d = d

    def _set_params_f01_alpha(self, f01, alpha, ng=0):
        """从实验测量的跃迁频率和非谐性反推 EJ 和 Ec。

        使用优化算法从目标 f01 和 alpha 值反推物理参数。
        适用于已知量子比特频率特性的情况。

        参数:
            f01 (float): 基态到第一激发态的跃迁频率，单位 GHz。
            alpha (float): 非谐性（f12 - f01），通常为负值，单位 GHz。
            ng (float, 可选): 电荷偏置，默认为 0。
        """
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
        """从磁通可调量子比特的极值频率反推参数。

        对于具有不对称 SQUID 的磁通可调 Transmon，
        使用 flux=0（最大频率）和 flux=0.5（最小频率）处的
        频率和非谐性来反推 EJS、Ec 和 d。

        参数:
            f01_max (float): flux=0 时的最大跃迁频率，单位 GHz。
            f01_min (float): flux=0.5 时的最小跃迁频率，单位 GHz。
            alpha1 (float): flux=0 时的非谐性，单位 GHz。
            alpha2 (float, 可选): flux=0.5 时的非谐性，默认为 alpha1。
            ng (float, 可选): 电荷偏置，默认为 0。
        """
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

    def levels(self, flux=0, ng=0, N=5, select_range=None):
        """计算当前参数设置下的 Transmon 能级。

        使用对象的 Ec、EJ 和 d 参数计算在指定磁通量和电荷偏置下的能级。

        参数:
            flux (float, 可选): 磁通量，以 Φ₀ 为单位，默认为 0。
            ng (float, 可选): 电荷偏置，默认为 0。
            N (int, 可选): 计算的能级数，默认为 5。
            select_range (tuple, 可选): 返回的能级范围 (start, end)。

        返回:
            array: 本征值（能级），形状为 (N,)。
        """
        w, *_ = eig_singal_qubit(self.Ec,
                                 flux_to_EJ(flux, self.EJ, self.d),
                                 ng,
                                 levels=N,
                                 select_range=select_range,
                                 eigvals_only=True)
        return w

    @property
    def EJ1_EJ2(self):
        """两个约瑟夫森结的能量比值。

        对于不对称 SQUID，返回两个结的能量比：
        EJ1 / EJ2 = (1 + d) / (1 - d)

        返回:
            float: 约瑟夫森能量比值 EJ1/EJ2。
        """
        return (1 + self.d) / (1 - self.d)

    def chargeParityDiff(self, flux=0, ng=0, k=0):
        """计算电荷宇称差异。

        比较 ng 和 ng+0.5 处第 k 能级跃迁频率的差异，
        反映系统对电荷噪声的敏感度。这是评估 Transmon
        对电荷噪声抑制程度的重要指标。

        参数:
            flux (float, 可选): 磁通量，以 Φ₀ 为单位，默认为 0。
            ng (float, 可选): 电荷偏置基准值，默认为 0。
            k (int, 可选): 能级索引，默认为 0（基态到第一激发态）。

        返回:
            float: 电荷宇称差异，单位 GHz。值越小表示对电荷噪声越不敏感。

        公式:
            Δf = [f_k(ng) - f_k(ng+0.5)]
            其中 f_k 是第 k 能级跃迁频率。
        """
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
                    levels=5,
                    select_range=None):
    """计算 Transmon 量子比特能级的便捷函数。

    支持通过 x 值、周期和偏移量来计算对应的能级，适用于拟合实验数据。

    参数:
        x (float or array): 输入值（可以是磁通量扫描的原始值）。
        period (float): 周期，将 x 映射到磁通量单位。
        offset (float): 偏移量，用于校正零点。
        EJS (float): 对称约瑟夫森能量，单位 GHz。
        Ec (float): 充电能量，单位 GHz。
        d (float): 非对称参数。
        ng (float, 可选): 电荷偏置，默认为 0。
        levels (int, 可选): 计算的能级数，默认为 5。
        select_range (tuple, 可选): 返回的能级范围，默认为 None（计算前 levels 个能级）。

    返回:
        array: 本征值（能级），形状为 (levels,)。

    公式:
        flux = (x - offset) / period
    """
    q = Transmon(EJ=EJS, Ec=Ec, d=d)
    return q.levels(flux=(x - offset) / period,
                    ng=ng,
                    N=levels,
                    select_range=select_range)


def Rn_to_EJ(Rn, gap=200, T=10):
    """从正常态电阻计算约瑟夫森能量。

    使用 Ambegaokar-Baratoff 公式（包含温度修正）计算约瑟夫森能量。
    这是从约瑟夫森结的正常态电阻估算 EJ 的标准方法。

    参数:
        Rn (float): 正常态电阻，单位欧姆 (Ω)。
        gap (float, 可选): 超导能隙，单位微电子伏特 (μeV)，默认为 200 μeV。
            对于铝结，典型值约为 180-200 μeV。
        T (float, 可选): 温度，单位毫开尔文 (mK)，默认为 10 mK。

    返回:
        float: 约瑟夫森能量 EJ，单位 GHz。

    公式:
        Δ = gap × e (转换为焦耳)
        Ic = (πΔ / 2eRn) × tanh(Δ / 2kT)
        EJ = ℏIc / 2e / h = Icℏ / 2e / h = IcΦ₀/(2π) / h

    参考:
        Ambegaokar, V. & Baratoff, A. (1963). Tunneling Between Superconductors.
    """
    from scipy.constants import e, h, hbar, k, pi

    Delta = gap * e * 1e-6
    Ic = pi * Delta / (2 * e * Rn * RESISTANCE_UNIT) * np.tanh(
        Delta / (2 * k * T * 1e-3))
    EJ = Ic * hbar / (2 * e)
    return EJ / h / FREQ_UNIT


def flux_to_EJ(flux, EJS, d=0):
    """根据磁通量计算有效约瑟夫森能量。

    对于具有不对称 SQUID 的磁通可调量子比特，
    有效约瑟夫森能量随磁通量周期性变化。

    参数:
        flux (float or array): 磁通量，以磁通量子 Φ₀ 为单位。
        EJS (float): 对称约瑟夫森能量（flux=0 时的有效 EJ），单位 GHz。
        d (float, 可选): 非对称参数，默认为 0（对称）。
            两个结的能量比：EJ1/EJ2 = (1 + d) / (1 - d)。

    返回:
        float or array: 给定磁通量下的有效约瑟夫森能量，单位 GHz。

    公式:
        EJ(flux) = EJS × √(cos²(π·flux) + d²·sin²(π·flux))

    物理意义:
        - flux=0: EJ = EJS（最大值）
        - flux=0.5: EJ = EJS × |d|（最小值，不为零除非对称）
    """
    F = np.pi * flux
    EJ = EJS * np.sqrt(np.cos(F)**2 + d**2 * np.sin(F)**2)
    return EJ


def n_op(N=5):
    """创建电荷数算符矩阵。

    在电荷基 {|n⟩} 下，电荷数算符 n̂ 是对角矩阵，
    对角元为电荷数 n = -N, -N+1, ..., N。

    参数:
        N (int, 可选): 电荷数范围，默认为 5。
            矩阵维度为 (2N+1) × (2N+1)。

    返回:
        numpy.ndarray: 电荷数算符矩阵，形状为 (2N+1, 2N+1)。

    公式:
        n̂ = Σ_n n|n⟩⟨n|

    示例:
        >>> n_op(N=2)
        array([[-2,  0,  0,  0,  0],
               [ 0, -1,  0,  0,  0],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  1,  0],
               [ 0,  0,  0,  0,  2]])
    """
    return np.diag(np.arange(-N, N + 1))


def cos_phi_op(N=5):
    """创建相位余弦算符 cos(φ̂) 的矩阵表示。

    在电荷基 {|n⟩} 下，cos(φ̂) = (e^(iφ̂) + e^(-iφ̂))/2。
    由于 e^(±iφ̂)|n⟩ = |n±1⟩，因此该算符只在次对角线上有非零元。

    参数:
        N (int, 可选): 电荷数范围，默认为 5。
            矩阵维度为 (2N+1) × (2N+1)。

    返回:
        numpy.ndarray: cos(φ̂) 算符矩阵，形状为 (2N+1, 2N+1)。

    公式:
        cos(φ̂) = (1/2)Σ_n (|n⟩⟨n+1| + |n+1⟩⟨n|)

    物理意义:
        该算符对应于约瑟夫森能量项 -EJ·cos(φ̂)，
        描述 Cooper 对在结两侧的隧穿过程。
    """
    from scipy.sparse import diags
    return diags([np.full(
        (2 * N, ), 0.5), np.full((2 * N, ), 0.5)], [1, -1]).toarray()


def sin_phi_op(N=5):
    """创建相位正弦算符 sin(φ̂) 的矩阵表示。

    在电荷基 {|n⟩} 下，sin(φ̂) = (e^(iφ̂) - e^(-iφ̂))/(2i)。
    该算符是厄米的，在次对角线上有纯虚数矩阵元。

    参数:
        N (int, 可选): 电荷数范围，默认为 5。
            矩阵维度为 (2N+1) × (2N+1)。

    返回:
        numpy.ndarray: sin(φ̂) 算符矩阵，形状为 (2N+1, 2N+1)。

    公式:
        sin(φ̂) = (1/2i)Σ_n (|n⟩⟨n+1| - |n+1⟩⟨n|)

    注意:
        该算符常与电流算符相关，I = Ic·sin(φ̂)。
    """
    from scipy.sparse import diags
    return diags([np.full(
        (2 * N, ), 0.5j), np.full((2 * N, ), -0.5j)], [1, -1]).toarray()


def phi_op(N=5):
    """创建相位算符 φ̂ 的矩阵表示。

    使用快速傅里叶变换（FFT）在电荷基下构造相位算符。
    由于相位和电荷是共轭变量，满足 [φ̂, n̂] = i，
    相位算符在电荷基下没有简单的解析形式。

    参数:
        N (int, 可选): 电荷数范围，默认为 5。
            矩阵维度为 (2N+1) × (2N+1)。

    返回:
        numpy.ndarray: 相位算符 φ̂ 矩阵，形状为 (2N+1, 2N+1)。

    方法:
        1. 在相位基下，φ̂ 是对角矩阵
        2. 使用 FFT 将算符从相位基转换到电荷基

    注意:
        由于周期性边界条件，相位算符的定义存在一定任意性。
        对于小 N 值，该算符可能不是完全厄米的。
    """
    from scipy.fft import fft, ifft, ifftshift

    k = ifftshift(np.arange(-N, N + 1) * np.pi / N)
    psi = np.eye(k.shape[0])
    T = fft(psi, overwrite_x=True)
    T *= k
    return ifft(T, overwrite_x=True)


def Ec_matrix_from_capacitance(C, nodes, junctions):
    """从电容矩阵计算充电能量矩阵。

    将电路的电容矩阵转换为约瑟夫森结处的有效充电能量矩阵。
    这是多量子比特系统耦合分析的重要步骤。

    参数:
        C (array): 电容矩阵，单位飞法拉 (fF)。
            C[i,j] 表示节点 i 和 j 之间的电容，C[i,i] 表示节点 i 对地的总电容。
        nodes (list): 节点标识符列表。
        junctions (list): 结列表，每个元素是一个元组 (a, b)。

    返回:
        numpy.ndarray: 充电能量矩阵 Ec，单位 GHz。
            返回的矩阵大小为 (len(junctions), len(junctions))。

    公式:
        1. 将电容矩阵求逆得到电感矩阵: C⁻¹
        2. 计算 Ec = (2e²/h) × C⁻¹ (转换为频率单位)
        3. 使用关联矩阵 S 变换到适当的坐标系: Ec' = Vᵀ @ Ec @ V
           其中 V = S⁻¹

    物理意义:
        对角元 Ec[i,i] 是第 i 个结的充电能量。
        非对角元 Ec[i,j] 表示结 i 和 j 之间的电容耦合强度。
    """
    from scipy.constants import e, h

    # unit: GHz
    Ec = np.linalg.inv(C * CAP_UNIT) * 2 * e**2 / h / FREQ_UNIT
    S = complete_incidence_matrix(nodes, junctions)
    V = np.linalg.inv(S)
    Ec = V.T @ Ec @ V
    return Ec[:len(junctions), :len(junctions)]


def eig_singal_qubit(Ec,
                     EJ,
                     ng=0.0,
                     levels=5,
                     eps=1e-6,
                     select_range=None,
                     eigvals_only=False):
    """计算单量子比特的本征值和本征向量（自适应网格）。

    使用自适应电荷基大小来精确计算单 Transmon 量子比特的本征问题。
    从较小的电荷基开始，逐步增大直到结果收敛。

    参数:
        Ec (float): 充电能量，单位 GHz。
        EJ (float): 约瑟夫森能量，单位 GHz。
        ng (float, 可选): 电荷偏置，默认为 0。
        levels (int, 可选): 需要计算的能级数，默认为 5。
        eps (float, 可选): 收敛精度，默认为 1e-6。
        select_range (tuple, 可选): 返回的能级范围 (start, end)，默认为 None（计算所有能级）。
        eigvals_only (bool, 可选): 是否只计算本征值，默认为 False。

    返回:
        tuple: (w, n, v, N)
            - w (array): 本征值（能级），形状为 (levels,)。
            - n (array): 电荷算符的期望值 n̂ 在本征态下的矩阵，形状为 (levels, levels)。如果 eigvals_only=True，则返回 None。
            - v (array): 本征向量矩阵，形状为 (2N+1, levels)。如果 eigvals_only=True，则返回 None。
            - N (int): 最终使用的电荷基大小。

    公式:
        哈密顿量: H = 4Ec(n - ng)² - (EJ/2)(|n⟩⟨n+1| + |n+1⟩⟨n|)

    算法:
        1. 从 N = levels + 2 开始构建哈密顿量
        2. 使用 scipy.linalg.eigh_tridiagonal 高效求解
        3. 增大 N 并重新计算，直到能级变化小于 eps
    """
    E = None
    if select_range is not None:
        levels = select_range[1]

    for N in range(levels + 2, 100):
        n = np.arange(-N, N + 1) - ng
        Hc = 4 * Ec * n**2
        HJ = -EJ * np.full(2 * N, 0.5)
        if select_range is not None:
            w = eigvalsh_tridiagonal(Hc,
                                     HJ,
                                     select='i',
                                     select_range=select_range)
        else:
            w = eigvalsh_tridiagonal(Hc, HJ)
            w = w[:levels]
        if E is not None:
            if np.all(np.abs(E - w) < eps):
                break
        E = w
    if eigvals_only:
        return w, None, None, N

    if select_range is not None:
        w, v = eigh_tridiagonal(Hc, HJ, select='i', select_range=select_range)
    else:
        w, v = eigh_tridiagonal(Hc, HJ)
        v = v[:, :levels]
        w = w[:levels]

    return w, v.T.conj() @ np.diag(n) @ v, v, N


def spectrum(Ec=100.0,
             EJS=6000.0,
             flux=0.0,
             ng=0.0,
             d=0.0,
             levels=5,
             decouple=False,
             return_psi=False):
    """计算多量子比特耦合系统的能谱。

    计算具有电容耦合的多 Transmon 量子比特系统的本征能谱。
    可以处理磁通可调量子比特和非对称 SQUID 的情况。

    参数:
        Ec (float or array): 充电能量。
            - 标量：单量子比特情况
            - 对角矩阵 (n,n)：多量子比特，对角元为各量子比特的充电能量
            - 完整矩阵 (n,n)：包含非对角耦合项
            单位与 EJS 一致即可，默认 100.0 (MHz)。
        EJS (float or array): 对称约瑟夫森能量，单位与 Ec 一致即可，默认 6000.0 (MHz)。
            可以是标量（所有量子比特相同）或数组（各量子比特不同）。
        flux (float or array): 磁通量，以 Φ₀ 为单位，默认 0.0。
            可以是标量或数组（各量子比特不同磁通偏置）。
        ng (float, 可选): 电荷偏置，默认为 0。
        d (float or array, 可选): 非对称参数，默认为 0。
            可以是标量或数组（各量子比特不同）。
        levels (int, 可选): 每个量子比特计算的能级数，默认为 5。
        decouple (bool, 可选): 是否忽略耦合计算，默认为 False。
            如果为 True，只返回非耦合能级（用于对比）。
        return_psi (bool, 可选): 是否返回本征波函数，默认为 False。

    返回:
        array: 能谱（相对于基态的能量差），单位与 Ec 和 EJS 一致。
            如果 return_psi=False，返回形状为 (num_levels,) 的能谱。
        或 tuple: (spectrum, psi, Hc) 如果 return_psi=True
            - spectrum: 能谱数组
            - psi: 本征波函数矩阵
            - Hc: 耦合哈密顿量矩阵

    哈密顿量:
        H = H₀ + Hc
        H₀ = Σᵢ Hᵢ (各量子比特的无耦合哈密顿量)
        Hc = Σᵢⱼ 8Ec[i,j] × nᵢ ⊗ nⱼ (电容耦合项)

    示例:
        >>> # 单量子比特
        >>> spec = spectrum(Ec=0.2, EJS=20)
        >>> spec[:2]  # 前两个跃迁频率
        array([4.963..., 4.763...])

        >>> # 两耦合量子比特
        >>> Ec_mat = [[0.2, 0.01], [0.01, 0.25]]  # 包含耦合
        >>> spec = spectrum(Ec_mat, EJS=[20, 18])
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
