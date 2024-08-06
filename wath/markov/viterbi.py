import numpy as np


def hmm_viterbi(z, QP_tau=1e-3, Pg=0.999, Pe=0.98, shot_time=3.2e-6):
    """
    Viterbi 算法
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
    print("Observations:", obs)
    print("Most likely hidden states:", path)
    return path
