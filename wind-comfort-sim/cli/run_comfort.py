import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import welch
from models.mdof import shear_building, modal_analysis
from models.rayleigh import rayleigh_coeffs, C_rayleigh
from solve.newmark import newmark_lin
from loads.ar_process import ar1_colored_noise
from comfort.rms_metrics import sliding_rms
from comfort.iso10137 import load_iso_config, get_bands

def simulate_once(n, m, k, zeta, dt, T, tau_s, S, seed):
    # structure
    M, K = shear_building(n, m, k)         # ensure mdof returns float
    M = M.astype(float); K = K.astype(float)
    f, w, phi = modal_analysis(M, K)
    a0, a1 = rayleigh_coeffs(w[0], w[1], zeta)
    C = C_rayleigh(M, K, a0, a1)
    # wind
    nt = int(T/dt); fs = 1.0/dt
    pattern = (np.diag(M).astype(float) * phi[:,0].astype(float))
    pattern = pattern / pattern.max()
    x = ar1_colored_noise(nt, dt, tau_s, sigma=1.0, seed=seed).astype(float)
    F = float(S) * np.outer(pattern, x)  # S must be float
    # integrate
    u,v,a = newmark_lin(M, C, K, F, dt)
    atop = a[-1,:]                 # m/s^2
    mg   = 1000.0 * atop / 9.80665 # milli-g
    return mg, fs, f[0]            # return top-floor accel in milli-g, sampling rate, f1

def tornado(n, m, k, zeta, dt, T, tau_s, S, seed, win_sec):
    """±10% finite-diff sensitivity of RMS to m,k,zeta."""
    base_mg, fs, _ = simulate_once(n,m,k,zeta,dt,T,tau_s,S,seed)
    base_rms = float(np.sqrt(np.mean(base_mg**2)))
    out = []
    for name, val, up, dn in [
        ("mass m", m, 1.10*m, 0.90*m),
        ("stiffness k", k, 1.10*k, 0.90*k),
        ("damping ζ", zeta, 1.10*zeta, 0.90*zeta),
    ]:
        mg_up, _, _ = simulate_once(n, (up if name=="mass m" else m),
                                       (up if name=="stiffness k" else k),
                                       (up if name=="damping ζ" else zeta),
                                       dt,T,tau_s,S,seed)
        mg_dn, _, _ = simulate_once(n, (dn if name=="mass m" else m),
                                       (dn if name=="stiffness k" else k),
                                       (dn if name=="damping ζ" else zeta),
                                       dt,T,tau_s,S,seed)
        rms_up = float(np.sqrt(np.mean(mg_up**2)))
        rms_dn = float(np.sqrt(np.mean(mg_dn**2)))
        out.append((name, rms_up-base_rms, base_rms-rms_dn))
    # convert to absolute deltas for tornado bars
    return base_rms, out

def main(cfg_path, iso_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    iso_cfg = load_iso_config(iso_path)
    bands, win_sec_iso = get_bands(iso_cfg)

    # STRICTLY cast types from YAML
    n    = int(cfg["n_stories"])
    m    = float(cfg["mass_per_floor"])
    k    = float(cfg["k_story"])
    zt   = float(cfg["zeta_target"])
    dt   = float(cfg.get("dt", 0.02))
    T    = float(cfg.get("duration_s", 3600))
    tau  = float(cfg.get("wind_tau_s", 20.0))
    S    = float(cfg.get("wind_scale", 1.0))
    seed = int(cfg.get("seed", 42))

    # --- simulate several zeta values for overlay ---
    zetas = cfg.get("zeta_sweep", [0.005, 0.01, 0.02])
    results = []
    plt.figure(figsize=(6,4))
    # draw comfort bands
    y0 = 0
    for i,(lab,maxmg,col) in enumerate(bands):
        plt.axhspan(y0, maxmg, color=(col if col else "#eee"), alpha=0.35, label=(lab if i==0 else None))
        y0 = maxmg

    for z in zetas:
        mg, fs, f1 = simulate_once(n,m,k,z,dt,T,tau,S,seed)
        rms_win = sliding_rms(mg, fs, win_sec_iso) # windowed RMS per ISO
        plt.plot(np.arange(len(rms_win))/fs/60.0, rms_win, label=f"ζ={100*z:.1f}%")

    plt.xlabel("Time (minutes)")
    plt.ylabel("RMS accel (milli-g)")
    plt.title(f"ISO overlay (window={int(win_sec_iso)} s), f1≈{f1:.3f} Hz")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs\\iso_overlay_rms.png", dpi=200)

    # --- tornado sensitivity at nominal zeta ---
    base_rms, sens = tornado(n,m,k,zt,dt,T,tau,S,seed, win_sec_iso)
    # format bars
    labels = [s[0] for s in sens]
    ups    = [max(0.0, s[1]) for s in sens]
    dns    = [max(0.0, s[2]) for s in sens]

    plt.figure(figsize=(5,3.5))
    y = np.arange(len(labels))
    plt.barh(y, ups, left=base_rms, alpha=0.7, label="+10%")
    plt.barh(y, [-d for d in dns], left=base_rms, alpha=0.7, label="-10%")
    plt.axvline(base_rms, color="k", linestyle="--", linewidth=1)
    plt.yticks(y, labels)
    plt.xlabel("RMS accel (milli-g)")
    plt.title(f"Tornado @ ζ={100*zt:.1f}% (base RMS={base_rms:.1f} mg)")
    plt.tight_layout()
    plt.savefig("figs\\tornado_rms.png", dpi=200)

    # dump a small summary CSV
    pd.DataFrame({
        "zeta": zetas
    }).to_csv("data\\comfort_summary.csv", index=False)

    print("Saved: figs\\iso_overlay_rms.png, figs\\tornado_rms.png, data\\comfort_summary.csv")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs\\base.yaml")
    ap.add_argument("--iso",    default="configs\\iso10137_office.yaml")
    args = ap.parse_args()
    main(args.config, args.iso)
