import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from models.mdof import shear_building, modal_analysis
from models.rayleigh import rayleigh_coeffs, C_rayleigh
from solve.newmark import newmark_lin
from loads.ar_process import ar1_colored_noise


def main(cfg_path):
    print(f"[run] cfg={cfg_path}")
    if not os.path.exists(cfg_path):
        print("[error] config file not found")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Defaults (shorter run for quick sanity)
    n = int(cfg.get("n_stories", 10))
    m = float(cfg.get("mass_per_floor", 1.0))
    k = float(cfg.get("k_story", 1.0e6))
    zt = float(cfg.get("zeta_target", 0.01))
    dt = float(cfg.get("dt", 0.02))
    T = float(cfg.get("duration_s", 600.0))
    tau = float(cfg.get("wind_tau_s", 20.0))
    S = float(cfg.get("wind_scale", 1.0))
    seed = int(cfg.get("seed", 42))

    # auto options
    f1_target = float(cfg.get("f1_target_hz", 0.20))
    auto_dt = bool(cfg.get("auto_dt", True))

    print(f"[cfg] n={n}, m={m}, k={k}, zeta={zt}, dt={dt}, T={T}, tau={tau}, S={S}, seed={seed}")

    # structure
    M, K = shear_building(n, m, k)
    f, w, phi = modal_analysis(M, K)

    # --- scale K to hit target f1 (if requested) ---
    if f[0] > 0 and f1_target > 0:
        scaleK = (f1_target / f[0]) ** 2
        K = K * scaleK
        f, w, phi = modal_analysis(M, K)

    # damping after any scaling
    a0, a1 = rayleigh_coeffs(w[0], w[1], zeta=zt)
    C = C_rayleigh(M, K, a0, a1)

    # eigen info + Newmark stability hint
    print(f"[eig] f1={f[0]:.3f} Hz, f2={f[1]:.3f} Hz | Rayleigh a0={a0:.3e}, a1={a1:.3e}")
    fmax = float(f[-1])
    if fmax > 0:
        dt_rec = 1.0 / (40.0 * fmax)
        print(f"[stability] fmax={fmax:.3f} Hz → recommend dt ≤ {dt_rec:.4f} s (current dt={dt:.4f})")
        if auto_dt:
            dt_acc = 1.0 / (50.0 * fmax)
            if dt_acc < dt:
                print(f"[time-step] auto reducing dt from {dt:.5f}s to {dt_acc:.5f}s (fmax={fmax:.3f} Hz)")
                dt = float(dt_acc)

    # wind force time history
    nt = int(T / dt)
    print(f"[time] nt={nt} steps")

    # floor force pattern ∝ mass × mode-1 shape (along-wind proxy)
    pattern = (np.diag(M) * phi[:, 0]).astype(float)
    pattern /= np.max(np.abs(pattern))  # unit peak per floor

    # AR(1) colored wind scalar, then distribute by pattern
    x = ar1_colored_noise(nt, dt, tau, sigma=1.0, seed=seed)
    F = S * np.outer(pattern, x)  # (n, nt)

    # Optional one-shot auto-tuning (quick helper) — purely for testing/calibration.
    # Tries a few S (force scale) values, reports RMS top-floor accel (milli-g),
    # and breaks when RMS >= 0.5 * target (so you can pick a value near 5–30 mg).
    target = 10.0  # mg
    for trial in [1e5, 3e5, 1e6, 3e6, 1e7]:
        S = trial
        F = S * np.outer(pattern, x)
        u_t, v_t, a_t = newmark_lin(M, C, K, F, dt)
        mg_t = 1000.0 * a_t[-1, :] / 9.80665
        rms = float(np.sqrt(np.mean(mg_t**2)))
        print(f"[calib] S={S:.1e} N → RMS={rms:.2f} mg")
        if rms >= 0.5 * target:
            # keep the S that met the threshold and its F for the main run
            F = S * np.outer(pattern, x)
            break

    # integrate
    u, v, a = newmark_lin(M, C, K, F, dt)

    # top-floor acceleration (in milli-g)
    atop = a[-1, :]
    mg = 1000.0 * atop / 9.80665

    # PSD (Welch) – guard nperseg
    nper = min(4096, len(mg))
    fw, Pw = welch(mg, fs=1 / dt, nperseg=nper)

    # ensure output dirs exist
    figs_dir = os.path.join("figs")
    data_dir = os.path.join("data")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    pd.DataFrame({"t": np.arange(nt) * dt, "a_top_mg": mg[:nt]}).to_csv(os.path.join(data_dir, "a_top_snippet.csv"), index=False)

    # (1) a(t) 60 s snippet
    Nsnip = min(int(60 / dt), len(mg))
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(Nsnip) * dt, mg[:Nsnip])
    plt.xlabel("Time (s)")
    plt.ylabel("Top accel (milli-g)")
    plt.title(f"a(t) top floor, ζ={zt*100:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "a_top_snippet.png"), dpi=200)

    # (2) PSD with f1 marker
    plt.figure(figsize=(6, 4))
    plt.semilogy(fw, Pw)
    plt.axvline(f[0], linestyle="--", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD [milli-g^2/Hz]")
    plt.title("Top accel PSD (Welch)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "a_top_psd.png"), dpi=200)

    rms_mg = float(np.sqrt(np.mean(mg**2)))
    print(f"[out] RMS(top) = {rms_mg:.2f} milli-g")
    print(f"Saved: {os.path.join(figs_dir,'a_top_snippet.png')}, {os.path.join(figs_dir,'a_top_psd.png')}, {os.path.join(data_dir,'a_top_snippet.csv')}")

    # extra sanity diagnostics (in-scope)
    print(f"[sanity] max|F|={np.max(np.abs(F)):.3e} N, "
        f"sum(pattern)={pattern.sum():.3e}, max|a|={np.max(np.abs(a)):.3e} m/s^2")


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default=os.path.join("configs", "base.yaml"))
        args = ap.parse_args()
        main(args.config)
    except Exception as e:
        print("[fatal]", repr(e))
        sys.exit(1)
