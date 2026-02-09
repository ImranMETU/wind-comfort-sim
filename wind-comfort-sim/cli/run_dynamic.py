import argparse, yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from models.mdof import shear_building, modal_analysis
from models.rayleigh import rayleigh_coeffs, C_rayleigh, modal_damping

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    n   = int(cfg["n_stories"])
    m   = float(cfg["mass_per_floor"])
    k   = float(cfg["k_story"])
    zt  = float(cfg["zeta_target"])

    # Optional debug: verify types parsed from config
    print(f"[cfg] n={n} ({type(n)}), m={m} ({type(m)}), k={k} ({type(k)}), zeta={zt} ({type(zt)})")

    M, K     = shear_building(n, m, k)
    print(f"[Kcheck] K00={K[0,0]:.3e}, K01={K[0,1]:.3e}, K11={K[1,1]:.3e}")
# Expect ~ K00 = 2k, K01 = -k, K11 = 2k  (with k = 1.0e6)
    f, w, ph = modal_analysis(M, K)

    a0, a1   = rayleigh_coeffs(w[0], w[1], zeta=zt)
    C        = C_rayleigh(M, K, a0, a1)
    z_eff    = modal_damping(C, M, K, ph, w)

    # Save modal table
    df = pd.DataFrame({"mode": np.arange(1, n+1), "f_Hz": f, "zeta_eff": z_eff})
    df.to_csv("data\\modal_table.csv", index=False)

    # Plot mode shapes
    stories = np.arange(1, n+1)
    plt.figure(figsize=(4,6))
    plt.plot(ph[:,0], stories, "o-", label=f"Mode 1 ({f[0]:.3f} Hz)")
    plt.plot(ph[:,1], stories, "s-", label=f"Mode 2 ({f[1]:.3f} Hz)")
    plt.gca().invert_yaxis()
    plt.xlabel("Normalized displacement")
    plt.ylabel("Story")
    plt.legend()
    plt.title(f"Mode Shapes (Rayleigh ζ1≈ζ2≈{zt*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("figs\\modes_1_2.png", dpi=200)
    print("Saved: data\\modal_table.csv, figs\\modes_1_2.png")
    print(f"Rayleigh a0={a0:.3e}, a1={a1:.3e}")
    print(f"ζ_eff(1)={z_eff[0]*100:.2f}%, ζ_eff(2)={z_eff[1]*100:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs\\base.yaml")
    args = ap.parse_args()
    main(args.config)
