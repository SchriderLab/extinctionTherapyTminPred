#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(
    "/users/w/a/wang0207/cancer_rescue/cancer_sims"
)

PAT = (
    ROOT /
    "fixed80_hybrid_heterogeneous_regenerated_v1"
)

ABC_DIR = (
    PAT /
    "abc_hybrid10k_4scan"
)

ABC_FILE = (
    ABC_DIR /
    "abc_80patients_4scan_hybrid10k.csv"
)

E_FILE = (
    PAT /
    "extinction_by_lag_binary300.csv"
)

OUT = (
    ABC_DIR /
    "regret_decomposition"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load
# ============================================================

abc = pd.read_csv(
    ABC_FILE
)

E = pd.read_csv(
    E_FILE
)

E["patient_id"] = (
    E["patient_id"]
    .astype(int)
)

E["lag"] = (
    E["lag"]
    .astype(int)
)


# ============================================================
# Extinction surface lookup
# ============================================================

lookup = {
    (
        int(r.patient_id),
        int(r.lag)
    ):
    float(
        r.extinction_probability
    )

    for r in E.itertuples()
}


oracle = (
    E.groupby(
        "patient_id"
    )
    ["extinction_probability"]
    .max()
    .to_dict()
)


rows = []


for r in abc.itertuples():

    pid = int(
        r.patient_id
    )

    method = str(
        r.method
    )

    # Primary discretization:
    # nearest predicted treatment cycle.
    pred_lag = int(
        r.pred_switch_lag_round
    )

    true_lag = int(
        r.true_switch_lag_at_tmin
    )

    E_oracle = float(
        oracle[pid]
    )

    E_tmin = lookup[
        (
            pid,
            true_lag
        )
    ]

    E_abc = lookup[
        (
            pid,
            pred_lag
        )
    ]

    total_regret = (
        E_oracle -
        E_abc
    )

    hidden_state_gap = (
        E_oracle -
        E_tmin
    )

    nadir_prediction_penalty = (
        E_tmin -
        E_abc
    )

    # Exact identity.
    assert np.isclose(
        total_regret,
        hidden_state_gap
        +
        nadir_prediction_penalty
    )

    rows.append({
        "patient_id":
            pid,

        "method":
            method,

        "k":
            int(r.k),

        "slope_weight":
            float(
                r.slope_weight
            ),

        "true_tmin":
            float(
                r.true_tmin
            ),

        "pred_tmin":
            float(
                r.pred_tmin_weighted
            ),

        "abs_tmin_error":
            float(
                r.abs_error_weighted
            ),

        "true_tmin_lag":
            true_lag,

        "abc_lag":
            pred_lag,

        "oracle_extinction":
            E_oracle,

        "tmin_extinction":
            E_tmin,

        "abc_extinction":
            E_abc,

        "total_oracle_regret":
            total_regret,

        # This is the loss remaining even if
        # t_min were known perfectly.
        "oracle_minus_true_tmin":
            hidden_state_gap,

        # Signed intentionally.
        #
        # Negative means ABC accidentally chose
        # a lag that beats switching at t_min.
        "true_tmin_minus_abc":
            nadir_prediction_penalty,
    })


D = pd.DataFrame(
    rows
)


D.to_csv(
    OUT /
    "abc_oracle_regret_by_patient.csv",
    index=False
)


# ============================================================
# Patient bootstrap
# ============================================================

def bootstrap_mean_ci(
    x,
    seed=1,
    B=20000
):

    x = np.asarray(
        x,
        dtype=float
    )

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        B
    )

    n = len(x)

    for b in range(B):

        sample = rng.choice(
            x,
            size=n,
            replace=True
        )

        means[b] = (
            sample.mean()
        )

    return (
        float(
            np.quantile(
                means,
                .025
            )
        ),
        float(
            np.quantile(
                means,
                .975
            )
        ),
    )


# ============================================================
# Summary by method
# ============================================================

summary_rows = []


for j, (
    method,
    g
) in enumerate(
    D.groupby(
        "method"
    )
):

    stats = {
        "method":
            method,

        "n":
            len(g),

        "mean_oracle_E":
            g[
                "oracle_extinction"
            ].mean(),

        "mean_true_tmin_E":
            g[
                "tmin_extinction"
            ].mean(),

        "mean_abc_E":
            g[
                "abc_extinction"
            ].mean(),

        "mean_total_regret":
            g[
                "total_oracle_regret"
            ].mean(),

        "mean_gap_oracle_to_true_tmin":
            g[
                "oracle_minus_true_tmin"
            ].mean(),

        "mean_nadir_prediction_penalty":
            g[
                "true_tmin_minus_abc"
            ].mean(),

        "median_total_regret":
            g[
                "total_oracle_regret"
            ].median(),

        "patients_zero_total_regret":
            int(
                np.isclose(
                    g[
                        "total_oracle_regret"
                    ],
                    0
                ).sum()
            ),

        "patients_abc_beats_tmin":
            int(
                (
                    g[
                        "true_tmin_minus_abc"
                    ]
                    <
                    0
                ).sum()
            ),

        "patients_abc_equals_tmin":
            int(
                np.isclose(
                    g[
                        "true_tmin_minus_abc"
                    ],
                    0
                ).sum()
            ),

        "patients_abc_worse_than_tmin":
            int(
                (
                    g[
                        "true_tmin_minus_abc"
                    ]
                    >
                    0
                ).sum()
            ),
    }


    for k, col in enumerate([
        "oracle_extinction",
        "tmin_extinction",
        "abc_extinction",
        "total_oracle_regret",
        "oracle_minus_true_tmin",
        "true_tmin_minus_abc",
    ]):

        lo, hi = bootstrap_mean_ci(
            g[col],
            seed=(
                20260910
                +
                j * 100
                +
                k
            )
        )

        stats[
            f"{col}_CI_lo"
        ] = lo

        stats[
            f"{col}_CI_hi"
        ] = hi


    summary_rows.append(
        stats
    )


S = pd.DataFrame(
    summary_rows
)

S.to_csv(
    OUT /
    "abc_oracle_regret_summary.csv",
    index=False
)


# ============================================================
# Primary method
# CV-selected BEFORE held-out patient evaluation
# ============================================================

primary = D[
    D[
        "method"
    ]
    ==
    "logN_plus_slope"
].copy()


print()
print("=" * 76)
print("PRIMARY: CV-SELECTED SLOPE-AWARE ABC")
print("=" * 76)

print(
    "patients:",
    len(primary)
)

print()

print(
    "mean oracle extinction:",
    primary[
        "oracle_extinction"
    ].mean()
)

print(
    "mean extinction at TRUE tmin:",
    primary[
        "tmin_extinction"
    ].mean()
)

print(
    "mean extinction at ABC predicted tmin:",
    primary[
        "abc_extinction"
    ].mean()
)

print()

print(
    "TOTAL ORACLE REGRET:",
    primary[
        "total_oracle_regret"
    ].mean()
)

print(
    "gap remaining with PERFECT tmin:",
    primary[
        "oracle_minus_true_tmin"
    ].mean()
)

print(
    "penalty from imperfect tmin prediction:",
    primary[
        "true_tmin_minus_abc"
    ].mean()
)

print()

print(
    "check decomposition:",
    primary[
        "oracle_minus_true_tmin"
    ].mean()
    +
    primary[
        "true_tmin_minus_abc"
    ].mean()
)

print()

print(
    "ABC beats true-tmin policy:",
    int(
        (
            primary[
                "true_tmin_minus_abc"
            ]
            <
            0
        ).sum()
    )
)

print(
    "ABC ties true-tmin policy:",
    int(
        np.isclose(
            primary[
                "true_tmin_minus_abc"
            ],
            0
        ).sum()
    )
)

print(
    "ABC worse than true-tmin policy:",
    int(
        (
            primary[
                "true_tmin_minus_abc"
            ]
            >
            0
        ).sum()
    )
)


print()
print(
    "=== BOTH METHODS ==="
)

print(
    S[
        [
            "method",
            "n",
            "mean_oracle_E",
            "mean_true_tmin_E",
            "mean_abc_E",
            "mean_total_regret",
            "mean_gap_oracle_to_true_tmin",
            "mean_nadir_prediction_penalty",
            "patients_zero_total_regret",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "=== LARGEST TOTAL REGRETS, PRIMARY METHOD ==="
)

print(
    primary.sort_values(
        "total_oracle_regret",
        ascending=False
    )
    [
        [
            "patient_id",
            "true_tmin",
            "pred_tmin",
            "true_tmin_lag",
            "abc_lag",
            "oracle_extinction",
            "tmin_extinction",
            "abc_extinction",
            "total_oracle_regret",
            "oracle_minus_true_tmin",
            "true_tmin_minus_abc",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print(
    "WROTE:",
    OUT /
    "abc_oracle_regret_by_patient.csv"
)

print(
    "WROTE:",
    OUT /
    "abc_oracle_regret_summary.csv"
)
