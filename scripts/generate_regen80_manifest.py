#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/users/w/a/wang0207/cancer_rescue/cancer_sims"
)

OUT = (
    ROOT /
    "fixed80_hybrid_heterogeneous_regenerated_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

N = 80
D = 5

rng = np.random.default_rng(20260908)

# ============================================================
# Latin hypercube without scipy dependency.
# Each parameter receives one sample from each 1/N interval.
# ============================================================

U = np.empty((N, D), dtype=float)

for j in range(D):
    perm = rng.permutation(N)
    U[:, j] = (
        perm + rng.random(N)
    ) / N


base_mut_rate = (
    1e-8 +
    U[:, 0] * (5e-8 - 1e-8)
)

dose_drug_1 = (
    0.60 +
    U[:, 1] * 0.20
)

turnover_rate = (
    0.10 +
    U[:, 2] * 0.20
)

draw_m2 = (
    0.25 +
    U[:, 3] * 0.40
)

draw_m3 = np.full(
    N,
    0.10
)

draw_m1 = (
    1.0 -
    draw_m2 -
    draw_m3
)

# ------------------------------------------------------------
# D2 variation.
#
# Dan's biological point: D2 should generally be weaker than D1.
#
# D2 minimum = 0.50
# D2 maximum = D1 - 0.05
#
# Thus:
# D1=0.60 -> D2 in ~[0.50,0.55]
# D1=0.80 -> D2 in ~[0.50,0.75]
# ------------------------------------------------------------

dose_drug_2 = (
    0.50 +
    U[:, 4] *
    (dose_drug_1 - 0.55)
)

df = pd.DataFrame({
    "patient_id":
        np.arange(N, dtype=int),

    "base_mut_rate":
        base_mut_rate,

    "dose_drug_1":
        dose_drug_1,

    "dose_drug_2":
        dose_drug_2,

    "turnover_rate":
        turnover_rate,

    "draw_m1":
        draw_m1,

    "draw_m2":
        draw_m2,

    "draw_m3":
        draw_m3,

    "cross_resistance":
        0.30,

    "expansion_time":
        20,

    # Initial expansion background.
    "expansion_seed":
        680000001 +
        np.arange(N, dtype=int),

    # SAME D1 stochastic continuation for every
    # lag/replicate belonging to this patient.
    "d1_seed":
        780000001 +
        np.arange(N, dtype=int),
})

assert df["patient_id"].nunique() == N
assert (df["dose_drug_2"] < df["dose_drug_1"]).all()
assert np.allclose(
    df["draw_m1"] +
    df["draw_m2"] +
    df["draw_m3"],
    1.0
)

path = OUT / "manifest_regenerated80.csv"

df.to_csv(
    path,
    index=False,
    float_format="%.12g"
)

print("WROTE:", path)
print()
print(df.describe().T)
print()
print(
    "D2 < D1 for all patients:",
    bool(
        (
            df["dose_drug_2"] <
            df["dose_drug_1"]
        ).all()
    )
)
