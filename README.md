# wind-comfort-sim
---

# Wind-Induced Acceleration and Comfort Evaluation

### Stochastic Wind Inputs + Rayleigh-Damped MDOF Building Model

This project simulates wind-induced motion of a tall building using a physically-consistent multi-degree-of-freedom (MDOF) dynamic model and evaluates occupant comfort using ISO-based perception criteria.

The goal is not structural safety assessment, but **serviceability performance**:

> How people feel motion — not whether the building collapses.

---

## Overview

Tall buildings under wind excitation can experience low-frequency sway perceptible to occupants.
Although rarely dangerous, such motion may cause discomfort, productivity loss, or complaints.

The workflow implemented in this repository:

1. Construct an MDOF shear-building dynamic model
2. Apply stochastic wind excitation
3. Compute acceleration time histories
4. Convert to perception-relevant metrics
5. Compare against ISO comfort thresholds
6. Study sensitivity to physical parameters

---

## Structural Model

The building is modeled as an **n-story shear structure**:

* Lumped mass at each floor
* Lateral stiffness per story
* Linear dynamic behavior

$$
\mathbf{M} = \mathrm{diag}(m_1, m_2, \dots, m_n)
$$


$$
\mathbf{K} = \text{banded nearest-neighbor stiffness matrix}
$$

Eigenvalue analysis yields modal frequencies and mode shapes.

### Damping Model

Rayleigh damping calibrated to match the first two modal damping ratios:

$$
\mathbf{C} = a_0 \mathbf{M} + a_1 \mathbf{K}
$$

Target damping:

> ζ ≈ 1% (typical ambient vibration for reinforced concrete towers)

---

### Mode Shapes

![Mode shapes](figs/modes_1_2.png)

### Modal Properties

| Mode | Frequency (Hz) | Effective Damping (%) |
| ---- | -------------- | --------------------- |
| 1    | ~0.20          | ~1.0                  |
| 2    | ~0.85          | ~1.0                  |

The first mode dominates top-floor lateral motion, consistent with real tall-building behavior.

---

## Wind Load Model

Wind excitation is modeled as a **stochastic colored-noise process**:

* AR(1) correlated time series
* Low-frequency dominant spectrum
* Distributed vertically using the first mode shape

This captures the frequency band relevant to human perception.

---

## Time Integration

Dynamic response is solved using the **Newmark-β method**:

$$
\gamma = \frac{1}{2}, \quad \beta = \frac{1}{4}
$$

Unconditionally stable for linear systems.

### Example Outputs

Top-floor acceleration (time history):
![Acc](figs/a_top_snippet.png)

Power spectral density:
![PSD](figs/a_top_psd.png)

A clear spectral peak appears near the fundamental frequency → sway-dominated response.

---

## Comfort Evaluation

Acceleration is converted to **milli-g** and processed using a **60-second sliding RMS window**.

Results are compared against ISO-style comfort perception bands for office occupancy.

`iso_overlay_rms.png`

### Interpretation

| Damping Ratio | Observed Comfort              |
| ------------- | ----------------------------- |
| 0.5%          | Occasional discomfort         |
| 1.0%          | Acceptable for most occupants |
| 2.0%          | Clearly comfortable           |

**Key insight:**
Damping controls perceived motion more than any other parameter.

---

## Sensitivity Study

A ±10% parametric sweep evaluates the influence of:

* Mass
* Stiffness
* Damping

`tornado_rms.png`

### Observed Trends

$$
\uparrow k \Rightarrow \downarrow a
$$
$$
\uparrow m \Rightarrow \uparrow a
$$
$$
\uparrow \zeta \Rightarrow \downarrow a
$$

Damping is the dominant mitigation parameter for wind comfort.

---

## Model Realism & Limitations

| Aspect           | Notes                                                    |
| ---------------- | -------------------------------------------------------- |
| Structural model | Shear-beam representation (no torsion)                   |
| Wind model       | AR(1) low-frequency surrogate (not full Kaimal spectrum) |
| Damping          | Constant viscous assumption                              |
| Comfort curves   | Office occupancy interpretation                          |

Despite simplifications, predicted RMS accelerations (5–30 milli-g) fall within documented tall-building ranges.

---

## Reproducibility

All results are generated from a **single configuration file** and fixed random seed.

### Environment

```
Python 3.12
numpy
scipy
matplotlib
pandas
pyyaml
openseespy (optional)
```

---

## Run Instructions

```bash
python cli/run_dynamic.py --config configs/base.yaml
python cli/run_timehist.py --config configs/base.yaml
python cli/run_comfort.py --config configs/base.yaml --iso configs/iso10137_office.yaml
python cli/run_sweep.py --config configs/base.yaml --plot
```

---

## Purpose

This repository provides a reproducible research framework for:

* Wind-induced building response simulation
* Human comfort evaluation
* Parameter sensitivity analysis

It serves as a foundation for integrating **structural dynamics, perception modeling, and serviceability-oriented SHM workflows**.
