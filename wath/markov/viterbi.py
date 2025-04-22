import numpy as np


def hmm_viterbi(z, QP_tau=1e-3, Pg=0.999, Pe=0.98, shot_time=3.2e-6):
    """
    Viterbi 算法，用来在给定一个观测序列 z 的前提下，求出最可能的隐藏状态序列。

    Parameters
    ----------
    z : list
        观测序列（长度为 T)，是一个整型数组，例如 [0,1,0,1,…], 每个元素表示在
        当前时间片 (shot) 观测到的符号 (0 或 1)。
    QP_tau : float
        两个隐藏状态之间转换的时间常数（以同一时间单位计）。它决定了状态在平均
        QP_tau 时间内完成一次跃迁的速率。
    Pg, Pe: float
        给定隐藏状态时，观测某个符号的概率。一般写成观测概率矩阵
    shot_time : float
        单个观测时间片的时长。结合 QP_tau 可以算出每个时间步的转移概率。
    """

    # 定义初始状态概率向量
    start_prob = np.log(np.array([1 - 1e-12, 1e-12]))

    # 定义转移概率矩阵   3.2us o--->e
    trans_rate_oe = 1 / QP_tau * shot_time

    trans_prob = np.log(
        np.array([[1 - trans_rate_oe, trans_rate_oe],
                  [trans_rate_oe, 1 - trans_rate_oe]]))

    # 定义观测概率矩阵
    # Pg = 0.972
    # Pe = 0.92
    obs_prob = np.log(np.array([[Pe, 1 - Pe], [1 - Pg, Pg]]))

    # 定义观测序列
    # obs = np.array([0, 1, 2, 0, 2, 1, 1, 0, 2, 1])
    obs = np.asarray(z)

    # 定义 Viterbi 算法的变量
    n_states = len(start_prob)
    T = len(obs)
    viterbi = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=np.int32)

    # 初始化 Viterbi 算法的第一列
    viterbi[:, 0] = start_prob + obs_prob[:, obs[0]]

    # 递推计算 Viterbi 算法的剩余部分
    for t in range(1, T):
        for s in range(n_states):
            # 计算每个状态的最大概率
            max_prob = viterbi[:, t - 1] + trans_prob[:, s] + obs_prob[s,
                                                                       obs[t]]
            # 更新 Viterbi 矩阵和后向指针矩阵
            viterbi[s, t] = np.max(max_prob)
            backpointer[s, t] = np.argmax(max_prob)

    # 回溯路径
    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(viterbi[:, -1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    # 打印结果
    # print("Observations:", obs)
    # print("Most likely hidden states:", path)
    return path
