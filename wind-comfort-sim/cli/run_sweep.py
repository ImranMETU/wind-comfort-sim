import argparse, yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from models.mdof import shear_building, modal_analysis
from models.rayleigh import rayleigh_coeffs, C_rayleigh
from solve.newmark import newmark_lin
from loads.ar_process import ar1_colored_noise

def simulate_once(n, m, k, zeta, dt, T, tau_s, S, seed):
    # structure
    M, K        = shear_building(int(n), float(m), float(k))
    f, w, phi   = modal_analysis(M, K)
    a0, a1      = rayleigh_coeffs(w[0], w[1], float(zeta))
    C           = C_rayleigh(M, K, a0, a1)

    # wind
    nt = int(float(T)/float(dt))
    pattern = (np.diag(M) * phi[:,0])
    pattern = pattern / np.max(pattern)
    x = ar1_colored_noise(nt, float(dt), float(tau_s), sigma=1.0, seed=int(seed))
    F = float(S) * np.outer(pattern, x)

    # integrate
    u,v,a = newmark_lin(M, C, K, F, float(dt))
    atop  = a[-1,:]                        # m/s^2
    mg    = 1000.0 * atop / 9.80665        # milli-g
    rms   = float(np.sqrt(np.mean(mg**2)))
    return dict(f1=float(f[0]), rms_mg=rms)

def main(cfg_path, out_csv, make_plot):
    # --- load & cast config safely ---
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    n    = int(cfg["n_stories"])
    m0   = float(cfg["mass_per_floor"])
    k0   = float(cfg["k_story"])
    z0   = float(cfg["zeta_target"])
    dt   = float(cfg.get("dt", 0.02))
    T    = float(cfg.get("duration_s", 3600))
    tau  = float(cfg.get("wind_tau_s", 20.0))
    S    = float(cfg.get("wind_scale", 1.0))
    seed = int(cfg.get("seed", 42))

    # --- 3x3x3 sweep grid (±10% around baseline) ---
    def pm10(x): return [0.90*float(x), float(x), 1.10*float(x)]
    m_levels   = pm10(m0)
    k_levels   = pm10(k0)
    z_levels   = pm10(z0)

    rows = []
    for m in m_levels:
        for k in k_levels:
            for z in z_levels:
                res = simulate_once(n, m, k, z, dt, T, tau, S, seed)
                rows.append({
                    "m": m, "k": k, "zeta": z,
                    "f1_Hz": res["f1"],
                    "rms_mg": res["rms_mg"]
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} with {len(df)} runs")

    if make_plot:
        # --- Tornado around baseline using direct recomputation ---
        base = simulate_once(n, m0, k0, z0, dt, T, tau, S, seed)
        base_rms = base["rms_mg"]

        # mass
        r_m_up = simulate_once(n, 1.10*m0, k0,      z0, dt, T, tau, S, seed)["rms_mg"]
        r_m_dn = simulate_once(n, 0.90*m0, k0,      z0, dt, T, tau, S, seed)["rms_mg"]
        # stiffness
        r_k_up = simulate_once(n, m0,      1.10*k0, z0, dt, T, tau, S, seed)["rms_mg"]
        r_k_dn = simulate_once(n, m0,      0.90*k0, z0, dt, T, tau, S, seed)["rms_mg"]
        # damping
        r_z_up = simulate_once(n, m0,      k0,      1.10*z0, dt, T, tau, S, seed)["rms_mg"]
        r_z_dn = simulate_once(n, m0,      k0,      0.90*z0, dt, T, tau, S, seed)["rms_mg"]

        labels = ["mass m", "stiffness k", "damping ζ"]
        ups    = [max(0.0, r_m_up - base_rms),
                  max(0.0, r_k_up - base_rms),
                  max(0.0, r_z_up - base_rms)]
        dns    = [max(0.0, base_rms - r_m_dn),
                  max(0.0, base_rms - r_k_dn),
                  max(0.0, base_rms - r_z_dn)]

        y = np.arange(len(labels))
        plt.figure(figsize=(5,3.6))
        plt.barh(y, ups, left=base_rms, alpha=0.75, label="+10%")
        plt.barh(y, [-d for d in dns], left=base_rms, alpha=0.75, label="-10%")
        plt.axvline(base_rms, color="k", linestyle="--", linewidth=1)
        plt.yticks(y, labels)
        plt.xlabel("RMS accel (milli-g)")
        plt.title(f"Tornado around baseline (RMS={base_rms:.1f} mg)")
        plt.tight_layout()
        plt.savefig("figs\\tornado_sweep.png", dpi=200)
        print("[OK] wrote figs\\tornado_sweep.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs\\base.yaml")
    ap.add_argument("--out", default="data\\sweep_runs.csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    main(args.config, args.out, args.plot)
