import numpy as np

def newmark_lin(M, C, K, F, dt, beta=1/4, gamma=1/2):
    M = np.array(M, dtype=float)
    C = np.array(C, dtype=float)
    K = np.array(K, dtype=float)
    F = np.array(F, dtype=float)

    n, nt = F.shape
    u = np.zeros((n, nt), dtype=float)
    v = np.zeros((n, nt), dtype=float)
    a = np.zeros((n, nt), dtype=float)

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    K_eff = K + a0 * M + a1 * C

    # initialize a0 from equilibrium
    a[:, 0] = np.linalg.solve(M, F[:, 0] - C @ v[:, 0] - K @ u[:, 0])

    for k in range(1, nt):
        P_eff = (F[:, k]
                 + M @ (a0 * u[:, k-1] + a2 * v[:, k-1] + a3 * a[:, k-1])
                 + C @ (a1 * u[:, k-1] + a4 * v[:, k-1] + a5 * a[:, k-1]))
        u[:, k] = np.linalg.solve(K_eff, P_eff)
        a[:, k] = a0 * (u[:, k] - u[:, k-1]) - a2 * v[:, k-1] - a3 * a[:, k-1]
        v[:, k] = v[:, k-1] + dt * ((1.0 - gamma) * a[:, k-1] + gamma * a[:, k])

    return u, v, a
