import numpy as np

def ar1_colored_noise(n, dt, tau, sigma=1.0, seed=42):
    """
    AR(1): x_t = rho x_{t-1} + e_t,  rho = exp(-dt/tau)
    tau ~ correlation time (s); sigma ~ target std of the process x_t
    """
    import numpy as _np
    rng  = _np.random.default_rng(int(seed))
    dt   = float(dt); tau = float(tau); sigma = float(sigma)
    # compute AR(1) coefficient, guarded
    rho = float(_np.exp(-dt / tau)) if tau > 0.0 else 0.0
    # innovation variance so that Var[x] = sigma^2 (clamp numerical noise)
    sigma_e = float(sigma * _np.sqrt(max(0.0, 1.0 - rho**2)))
    e = rng.normal(0.0, sigma_e, size=int(n)).astype(float)
    x = _np.zeros(int(n), dtype=float)
    for i in range(1, int(n)):
        x[i] = rho * x[i-1] + e[i]
    return x
