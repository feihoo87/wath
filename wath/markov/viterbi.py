import numpy as np


def viterbi_hmm(observations: np.ndarray,
                transition_time_constant_0_to_1: float = 1e-3,
                transition_time_constant_1_to_0: float = 1e-3,
                emission_prob_state0: float = 0.98,
                emission_prob_state1: float = 0.999,
                timestep: float = 3.2e-6) -> np.ndarray:
    """
    Perform the Viterbi algorithm for a two-state HMM with asymmetric time-based transition rates.

    Parameters
    ----------
    observations : array_like of int
        Sequence of observed symbols (0 or 1), shape (T,).
    transition_time_constant_0_to_1 : float, optional
        Average time between transitions from state 0 to state 1 (in same time unit as timestep).
    transition_time_constant_1_to_0 : float, optional
        Average time between transitions from state 1 to state 0 (in same time unit as timestep).
    emission_prob_state0 : float, optional
        Probability of correctly observing symbol 0 when the hidden state is 0,
        i.e., P(obs=0 | state=0) (true negative rate). Default 0.98.
    emission_prob_state1 : float, optional
        Probability of correctly observing symbol 1 when the hidden state is 1,
        i.e., P(obs=1 | state=1) (true positive rate). Default 0.999.
    timestep : float, optional
        Duration of each observation time step. Default 3.2e-6.

    Returns
    -------
    np.ndarray
        Most likely hidden state sequence (0 or 1), shape (T,).
    """
    obs = np.asarray(observations, dtype=int)
    T = obs.shape[0]
    N = 2  # number of hidden states

    # Initial state log-probabilities
    initial_log_prob = np.log(np.array([1 - 1e-12, 1e-12]))

    # Transition probabilities per timestep (asymmetric)
    rate_0_to_1 = timestep / transition_time_constant_0_to_1
    rate_1_to_0 = timestep / transition_time_constant_1_to_0
    log_transition = np.log(
        np.array([[1 - rate_0_to_1, rate_0_to_1],
                  [rate_1_to_0, 1 - rate_1_to_0]]))

    # Emission log-probabilities: rows=states, cols=observations
    log_emission = np.log(
        np.array([[emission_prob_state0, 1 - emission_prob_state0],
                  [1 - emission_prob_state1, emission_prob_state1]]))

    # Initialize Viterbi matrices
    viterbi = np.full((N, T), -np.inf)
    backpointer = np.zeros((N, T), dtype=int)

    # Initialization step
    viterbi[:, 0] = initial_log_prob + log_emission[:, obs[0]]

    # Recursion step
    for t in range(1, T):
        for s in range(N):
            prob = viterbi[:, t -
                           1] + log_transition[:, s] + log_emission[s, obs[t]]
            backpointer[s, t] = np.argmax(prob)
            viterbi[s, t] = prob[backpointer[s, t]]

    # Backtrace step
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(viterbi[:, -1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    return path


if __name__ == "__main__":
    # Example usage
    example_obs = [0, 1, 0, 1, 1, 0, 0, 1]
    hidden_states = viterbi_hmm(example_obs,
                                transition_time_constant_0_to_1=2e-3,
                                transition_time_constant_1_to_0=0.5e-3)
    print("Observations:", example_obs)
    print("Hidden state path:", hidden_states)
