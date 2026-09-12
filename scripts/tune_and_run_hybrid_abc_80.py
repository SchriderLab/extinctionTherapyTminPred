#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/users/w/a/wang0207/cancer_rescue/cancer_sims"
)

REF_DIR = (
    ROOT /
    "abc_ref_hybrid_regen80_v1_10k"
)

PAT_DIR = (
    ROOT /
    "fixed80_hybrid_heterogeneous_regenerated_v1"
)

OUT = (
    PAT_DIR /
    "abc_hybrid10k_4scan"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SCAN_TIMES = np.array(
    [0., 1., 2., 3.]
)

K_VALUES = [
    10,
    25,
    50,
    100,
    200,
    400,
]

SLOPE_WEIGHTS = [
    0.25,
    0.50,
    1.00,
    2.00,
]

N_FOLDS = 5

RANDOM_SEED = 20260910

BATCH_SIZE = 96

EPS = 1e-8


# ============================================================
# HELPERS
# ============================================================

def weighted_prediction(
    y,
    d
):

    w = 1.0 / (
        np.asarray(d, dtype=float)
        +
        EPS
    )

    y = np.asarray(
        y,
        dtype=float
    )

    return float(
        np.sum(w * y)
        /
        np.sum(w)
    )


def prediction_stats(
    y,
    d
):

    y = np.asarray(
        y,
        dtype=float
    )

    d = np.asarray(
        d,
        dtype=float
    )

    w = 1.0 / (
        d + EPS
    )

    wn = (
        w /
        w.sum()
    )

    return {
        "pred_tmin_weighted":
            float(
                np.sum(
                    wn * y
                )
            ),

        "pred_tmin_mean":
            float(
                np.mean(y)
            ),

        "pred_tmin_median":
            float(
                np.median(y)
            ),

        "posterior_q025":
            float(
                np.quantile(
                    y,
                    0.025
                )
            ),

        "posterior_q25":
            float(
                np.quantile(
                    y,
                    0.25
                )
            ),

        "posterior_q75":
            float(
                np.quantile(
                    y,
                    0.75
                )
            ),

        "posterior_q975":
            float(
                np.quantile(
                    y,
                    0.975
                )
            ),

        "best_distance":
            float(
                np.min(d)
            ),

        "mean_distance_k":
            float(
                np.mean(d)
            ),
    }


def summarize_predictions(
    x
):

    true = x[
        "true_tmin"
    ].to_numpy(float)

    pred = x[
        "pred_tmin_weighted"
    ].to_numpy(float)

    err = pred - true

    ae = np.abs(
        err
    )

    rounded = np.rint(
        pred
    )

    rounded_err = (
        rounded - true
    )

    return {
        "n":
            len(x),

        "bias_weighted":
            float(
                np.mean(err)
            ),

        "MAE_weighted":
            float(
                np.mean(ae)
            ),

        "median_AE_weighted":
            float(
                np.median(ae)
            ),

        "RMSE_weighted":
            float(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            ),

        "within_0p5":
            float(
                np.mean(
                    ae <= 0.5
                )
            ),

        "within_1":
            float(
                np.mean(
                    ae <= 1.0
                )
            ),

        "within_2":
            float(
                np.mean(
                    ae <= 2.0
                )
            ),

        "MAE_round":
            float(
                np.mean(
                    np.abs(
                        rounded_err
                    )
                )
            ),

        "exact_round":
            float(
                np.mean(
                    rounded_err == 0
                )
            ),

        "within_1_round":
            float(
                np.mean(
                    np.abs(
                        rounded_err
                    )
                    <= 1
                )
            ),
    }


def get_patient_prefix(
    pid
):

    pdir = (
        PAT_DIR /
        "patients" /
        f"patient_{pid:03d}"
    )

    candidates = []

    candidates.extend(
        sorted(
            pdir.glob(
                "*prefix*.csv"
            )
        )
    )

    candidates.extend(
        sorted(
            pdir.glob(
                "**/*prefix*.csv"
            )
        )
    )

    # Fallback only if naming differs.
    candidates.extend(
        sorted(
            pdir.glob(
                "*.csv"
            )
        )
    )

    candidates.extend(
        sorted(
            pdir.glob(
                "**/*.csv"
            )
        )
    )

    candidates = list(
        dict.fromkeys(
            candidates
        )
    )

    for f in candidates:

        if "outcome" in str(
            f
        ).lower():
            continue

        try:
            x = pd.read_csv(
                f,
                nrows=5
            )

        except Exception:
            continue

        if "N" not in x.columns:
            continue

        if any(
            c in x.columns
            for c in [
                "t_treatment",
                "t",
                "treatment_t",
                "cycle",
            ]
        ):
            return f

    raise FileNotFoundError(
        f"No usable prefix found "
        f"for patient {pid}: {pdir}"
    )


def extract_patient(
    pid,
    true_tmin
):

    f = get_patient_prefix(
        pid
    )

    x = pd.read_csv(
        f
    )

    tcol = None

    for c in [
        "t_treatment",
        "t",
        "treatment_t",
        "cycle",
    ]:

        if c in x.columns:
            tcol = c
            break

    if tcol is None:
        raise RuntimeError(
            f"No time column: {f}"
        )

    z = pd.DataFrame({
        "t":
            pd.to_numeric(
                x[tcol],
                errors="coerce"
            ),

        "N":
            pd.to_numeric(
                x["N"],
                errors="coerce"
            ),
    })

    z = (
        z
        .dropna()
        .sort_values("t")
        .drop_duplicates(
            "t",
            keep="last"
        )
    )

    # Only needed for a file using absolute cycle.
    if (
        tcol == "cycle"
        and
        z["t"].min() > 3
    ):

        z["t"] -= (
            z["t"].min()
        )

    Ns = []

    for t in SCAN_TIMES:

        q = z[
            np.isclose(
                z["t"],
                t,
                atol=1e-9
            )
        ]

        if len(q) != 1:

            raise RuntimeError(
                f"Patient {pid}: "
                f"missing exact scan t={t}"
            )

        Ns.append(
            float(
                q.iloc[0]["N"]
            )
        )

    idx = z[
        "N"
    ].idxmin()

    calculated_tmin = float(
        z.loc[
            idx,
            "t"
        ]
    )

    if not np.isclose(
        calculated_tmin,
        true_tmin
    ):

        raise RuntimeError(
            f"Patient {pid}: "
            f"summary tmin={true_tmin}, "
            f"prefix tmin={calculated_tmin}"
        )

    logN = np.log10(
        np.asarray(
            Ns,
            dtype=float
        )
        +
        1.0
    )

    return {
        "patient_id":
            pid,

        "file":
            str(f),

        "true_tmin":
            float(true_tmin),

        "true_switch_lag_at_tmin":
            int(
                round(
                    true_tmin
                )
            )
            +
            1,

        "N0":
            Ns[0],

        "N1":
            Ns[1],

        "N2":
            Ns[2],

        "N3":
            Ns[3],

        "logN0":
            logN[0],

        "logN1":
            logN[1],

        "logN2":
            logN[2],

        "logN3":
            logN[3],

        "dlog01":
            logN[1]
            -
            logN[0],

        "dlog12":
            logN[2]
            -
            logN[1],

        "dlog23":
            logN[3]
            -
            logN[2],
    }


def distance_matrices(
    train_level,
    train_slope,
    query_level,
    query_slope,
    sd_level,
    sd_slope
):

    level_dist = np.sqrt(
        np.mean(
            (
                (
                    query_level[
                        :,
                        None,
                        :
                    ]
                    -
                    train_level[
                        None,
                        :,
                        :
                    ]
                )
                /
                sd_level
            )
            ** 2,
            axis=2,
        )
    )

    slope_dist = np.sqrt(
        np.mean(
            (
                (
                    query_slope[
                        :,
                        None,
                        :
                    ]
                    -
                    train_slope[
                        None,
                        :,
                        :
                    ]
                )
                /
                sd_slope
            )
            ** 2,
            axis=2,
        )
    )

    return (
        level_dist,
        slope_dist
    )


def top_indices(
    distance,
    max_k
):

    max_k = min(
        max_k,
        len(distance)
    )

    idx = np.argpartition(
        distance,
        max_k - 1
    )[:max_k]

    idx = idx[
        np.argsort(
            distance[idx]
        )
    ]

    return idx


# ============================================================
# LOAD REFERENCE LIBRARY
# ============================================================

ref = pd.read_csv(
    REF_DIR /
    "abc_reference_features_usable.csv"
)

print(
    "usable reference file:",
    len(ref)
)


# ------------------------------------------------------------
# IMPORTANT:
# Exclude references whose apparent nadir is the final
# recorded observation. Their true nadir may lie later.
# ------------------------------------------------------------

if "tmin_boundary" in ref.columns:

    n_boundary = int(
        (
            pd.to_numeric(
                ref[
                    "tmin_boundary"
                ],
                errors="coerce"
            )
            ==
            1
        ).sum()
    )

else:

    n_boundary = 0


print(
    "excluding boundary-nadir refs:",
    n_boundary
)


if "tmin_boundary" in ref.columns:

    ref = ref[
        pd.to_numeric(
            ref[
                "tmin_boundary"
            ],
            errors="coerce"
        )
        !=
        1
    ].copy()


ref = ref.reset_index(
    drop=True
)


print(
    "clean ABC reference trajectories:",
    len(ref)
)


LEVEL_COLS = [
    "logN0",
    "logN1",
    "logN2",
    "logN3",
]


ref_level = (
    ref[
        LEVEL_COLS
    ]
    .to_numpy(float)
)

ref_slope = np.diff(
    ref_level,
    axis=1
)

ref_y = (
    ref[
        "t_min_true"
    ]
    .to_numpy(float)
)

ref_ids = (
    ref[
        "ref_id"
    ]
    .to_numpy(int)
)


# ============================================================
# LOAD 80 HELD-OUT PATIENTS
# ============================================================

patient_summary = pd.read_csv(
    PAT_DIR /
    "patient_oracle_characterization.csv"
)

patient_manifest = pd.read_csv(
    PAT_DIR /
    "manifest_regenerated80.csv"
)

patient_rows = []


for _, r in (
    patient_summary
    .sort_values(
        "patient_id"
    )
    .iterrows()
):

    pid = int(
        r[
            "patient_id"
        ]
    )

    patient_rows.append(
        extract_patient(
            pid,
            float(
                r["tmin"]
            )
        )
    )


patients = pd.DataFrame(
    patient_rows
)


assert len(
    patients
) == 80


patients = patients.merge(
    patient_manifest,
    on="patient_id",
    how="left",
    suffixes=(
        "",
        "_manifest"
    )
)


patient_level = (
    patients[
        LEVEL_COLS
    ]
    .to_numpy(float)
)

patient_slope = np.diff(
    patient_level,
    axis=1
)


print()
print(
    "held-out patients:",
    len(patients)
)

print(
    "patient tmin range:",
    patients[
        "true_tmin"
    ].min(),
    "to",
    patients[
        "true_tmin"
    ].max()
)


# ============================================================
# 5-FOLD REFERENCE-ONLY CV
#
# The 80 patients are NOT used for hyperparameter tuning.
# ============================================================

print()
print("=" * 72)
print("REFERENCE-ONLY 5-FOLD CROSS-VALIDATION")
print("=" * 72)


rng = np.random.default_rng(
    RANDOM_SEED
)

perm = rng.permutation(
    len(ref)
)

fold_id = np.empty(
    len(ref),
    dtype=int
)

fold_id[perm] = (
    np.arange(
        len(ref)
    )
    %
    N_FOLDS
)


cv_records = []

max_k_requested = max(
    K_VALUES
)


for fold in range(
    N_FOLDS
):

    train_idx = np.where(
        fold_id != fold
    )[0]

    val_idx = np.where(
        fold_id == fold
    )[0]

    Xl_train = ref_level[
        train_idx
    ]

    Xs_train = ref_slope[
        train_idx
    ]

    y_train = ref_y[
        train_idx
    ]

    Xl_val = ref_level[
        val_idx
    ]

    Xs_val = ref_slope[
        val_idx
    ]

    y_val = ref_y[
        val_idx
    ]

    ids_val = ref_ids[
        val_idx
    ]


    # Training-fold normalization only.
    sd_level = float(
        np.std(
            Xl_train.ravel()
        )
    )

    sd_slope = float(
        np.std(
            Xs_train.ravel()
        )
    )


    if (
        not np.isfinite(
            sd_level
        )
        or
        sd_level <= 0
    ):
        sd_level = 1.0


    if (
        not np.isfinite(
            sd_slope
        )
        or
        sd_slope <= 0
    ):
        sd_slope = 1.0


    print(
        f"fold {fold + 1}/{N_FOLDS}: "
        f"train={len(train_idx)} "
        f"validation={len(val_idx)} "
        f"sd_level={sd_level:.5f} "
        f"sd_slope={sd_slope:.5f}",
        flush=True
    )


    for start in range(
        0,
        len(val_idx),
        BATCH_SIZE
    ):

        stop = min(
            start + BATCH_SIZE,
            len(val_idx)
        )

        level_dist, slope_dist = (
            distance_matrices(
                Xl_train,
                Xs_train,
                Xl_val[
                    start:stop
                ],
                Xs_val[
                    start:stop
                ],
                sd_level,
                sd_slope
            )
        )


        for j in range(
            stop - start
        ):

            true_tmin = float(
                y_val[
                    start + j
                ]
            )

            ref_id_val = int(
                ids_val[
                    start + j
                ]
            )


            # =================================================
            # Baseline logN
            # =================================================

            d = level_dist[
                j
            ]

            order = top_indices(
                d,
                max_k_requested
            )


            for k in K_VALUES:

                use = order[
                    :k
                ]

                pred = weighted_prediction(
                    y_train[
                        use
                    ],
                    d[
                        use
                    ]
                )

                cv_records.append({
                    "fold":
                        fold,

                    "ref_id":
                        ref_id_val,

                    "method":
                        "baseline_logN",

                    "slope_weight":
                        0.0,

                    "k":
                        k,

                    "true_tmin":
                        true_tmin,

                    "pred_tmin_weighted":
                        pred,
                })


            # =================================================
            # logN + slopes
            # =================================================

            for sw in SLOPE_WEIGHTS:

                d = (
                    level_dist[
                        j
                    ]
                    +
                    sw
                    *
                    slope_dist[
                        j
                    ]
                )

                order = top_indices(
                    d,
                    max_k_requested
                )


                for k in K_VALUES:

                    use = order[
                        :k
                    ]

                    pred = weighted_prediction(
                        y_train[
                            use
                        ],
                        d[
                            use
                        ]
                    )

                    cv_records.append({
                        "fold":
                            fold,

                        "ref_id":
                            ref_id_val,

                        "method":
                            "logN_plus_slope",

                        "slope_weight":
                            sw,

                        "k":
                            k,

                        "true_tmin":
                            true_tmin,

                        "pred_tmin_weighted":
                            pred,
                    })


cv = pd.DataFrame(
    cv_records
)


cv_path = (
    OUT /
    "abc_reference_5fold_cv_predictions.csv"
)

cv.to_csv(
    cv_path,
    index=False
)


# ============================================================
# CV SUMMARY
# ============================================================

summary_rows = []


for (
    method,
    sw,
    k
), g in cv.groupby(
    [
        "method",
        "slope_weight",
        "k",
    ]
):

    stats = summarize_predictions(
        g
    )

    stats.update({
        "method":
            method,

        "slope_weight":
            float(sw),

        "k":
            int(k),
    })

    summary_rows.append(
        stats
    )


cv_summary = pd.DataFrame(
    summary_rows
)


cv_summary = cv_summary[
    [
        "method",
        "slope_weight",
        "k",
        "n",
        "bias_weighted",
        "MAE_weighted",
        "median_AE_weighted",
        "RMSE_weighted",
        "within_0p5",
        "within_1",
        "within_2",
        "MAE_round",
        "exact_round",
        "within_1_round",
    ]
]


cv_summary = cv_summary.sort_values(
    [
        "MAE_weighted",
        "RMSE_weighted",
        "k",
    ]
).reset_index(
    drop=True
)


cv_summary_path = (
    OUT /
    "abc_reference_5fold_cv_summary.csv"
)

cv_summary.to_csv(
    cv_summary_path,
    index=False
)


best_baseline = (
    cv_summary[
        cv_summary[
            "method"
        ]
        ==
        "baseline_logN"
    ]
    .sort_values(
        [
            "MAE_weighted",
            "RMSE_weighted",
        ]
    )
    .iloc[0]
)


best_slope = (
    cv_summary[
        cv_summary[
            "method"
        ]
        ==
        "logN_plus_slope"
    ]
    .sort_values(
        [
            "MAE_weighted",
            "RMSE_weighted",
        ]
    )
    .iloc[0]
)


print()
print("=== TOP 12 CV CONFIGURATIONS ===")

print(
    cv_summary
    .head(12)
    .to_string(
        index=False
    )
)


print()
print("=== BEST BASELINE ===")

print(
    best_baseline.to_string()
)


print()
print("=== BEST SLOPE-AWARE ===")

print(
    best_slope.to_string()
)


# ============================================================
# APPLY FROZEN SETTINGS TO 80 HELD-OUT PATIENTS
# ============================================================

print()
print("=" * 72)
print("HELD-OUT 80-PATIENT ABC")
print("=" * 72)


# Full-reference normalization.
sd_level_full = float(
    np.std(
        ref_level.ravel()
    )
)

sd_slope_full = float(
    np.std(
        ref_slope.ravel()
    )
)


configs = [
    {
        "method":
            "baseline_logN",

        "k":
            int(
                best_baseline[
                    "k"
                ]
            ),

        "slope_weight":
            0.0,
    },

    {
        "method":
            "logN_plus_slope",

        "k":
            int(
                best_slope[
                    "k"
                ]
            ),

        "slope_weight":
            float(
                best_slope[
                    "slope_weight"
                ]
            ),
    },
]


patient_predictions = []
neighbor_rows = []


for start in range(
    0,
    len(patients),
    BATCH_SIZE
):

    stop = min(
        start + BATCH_SIZE,
        len(patients)
    )

    level_dist, slope_dist = (
        distance_matrices(
            ref_level,
            ref_slope,
            patient_level[
                start:stop
            ],
            patient_slope[
                start:stop
            ],
            sd_level_full,
            sd_slope_full
        )
    )


    for j in range(
        stop - start
    ):

        prow = patients.iloc[
            start + j
        ]

        pid = int(
            prow[
                "patient_id"
            ]
        )

        true_tmin = float(
            prow[
                "true_tmin"
            ]
        )


        for cfg in configs:

            method = cfg[
                "method"
            ]

            k = cfg[
                "k"
            ]

            sw = cfg[
                "slope_weight"
            ]


            if method == "baseline_logN":

                d = level_dist[
                    j
                ]

            else:

                d = (
                    level_dist[
                        j
                    ]
                    +
                    sw
                    *
                    slope_dist[
                        j
                    ]
                )


            order = top_indices(
                d,
                k
            )

            use = order[
                :k
            ]

            stats = prediction_stats(
                ref_y[
                    use
                ],
                d[
                    use
                ]
            )

            pred = stats[
                "pred_tmin_weighted"
            ]

            pred_round = int(
                round(
                    pred
                )
            )

            pred_ceil = int(
                math.ceil(
                    pred
                )
            )

            # Key indexing rule:
            # prefix t=0 corresponds to outcome lag=1.
            switch_lag_round = (
                pred_round + 1
            )

            switch_lag_ceil = (
                pred_ceil + 1
            )

            switch_lag_round_clipped = int(
                np.clip(
                    switch_lag_round,
                    1,
                    20
                )
            )

            switch_lag_ceil_clipped = int(
                np.clip(
                    switch_lag_ceil,
                    1,
                    20
                )
            )


            outrow = {
                "patient_id":
                    pid,

                "method":
                    method,

                "k":
                    k,

                "slope_weight":
                    sw,

                "true_tmin":
                    true_tmin,

                "true_switch_lag_at_tmin":
                    int(
                        prow[
                            "true_switch_lag_at_tmin"
                        ]
                    ),

                "pred_tmin_weighted":
                    pred,

                "pred_tmin_mean":
                    stats[
                        "pred_tmin_mean"
                    ],

                "pred_tmin_median":
                    stats[
                        "pred_tmin_median"
                    ],

                "pred_tmin_round":
                    pred_round,

                "pred_tmin_ceil":
                    pred_ceil,

                "pred_switch_lag_round":
                    switch_lag_round_clipped,

                "pred_switch_lag_ceil":
                    switch_lag_ceil_clipped,

                "error_weighted":
                    pred
                    -
                    true_tmin,

                "abs_error_weighted":
                    abs(
                        pred
                        -
                        true_tmin
                    ),

                "abs_error_round":
                    abs(
                        pred_round
                        -
                        true_tmin
                    ),

                "posterior_q025":
                    stats[
                        "posterior_q025"
                    ],

                "posterior_q25":
                    stats[
                        "posterior_q25"
                    ],

                "posterior_q75":
                    stats[
                        "posterior_q75"
                    ],

                "posterior_q975":
                    stats[
                        "posterior_q975"
                    ],

                "posterior_ci95_width":
                    stats[
                        "posterior_q975"
                    ]
                    -
                    stats[
                        "posterior_q025"
                    ],

                "best_distance":
                    stats[
                        "best_distance"
                    ],

                "mean_distance_k":
                    stats[
                        "mean_distance_k"
                    ],

                "N0":
                    prow["N0"],

                "N1":
                    prow["N1"],

                "N2":
                    prow["N2"],

                "N3":
                    prow["N3"],

                "logN0":
                    prow[
                        "logN0"
                    ],

                "logN1":
                    prow[
                        "logN1"
                    ],

                "logN2":
                    prow[
                        "logN2"
                    ],

                "logN3":
                    prow[
                        "logN3"
                    ],
            }


            # Carry patient-generating parameters.
            for c in [
                "base_mut_rate",
                "dose_drug_1",
                "dose_drug_2",
                "turnover_rate",
                "draw_m1",
                "draw_m2",
                "draw_m3",
                "cross_resistance",
            ]:

                if c in prow.index:

                    outrow[
                        c
                    ] = prow[c]


            patient_predictions.append(
                outrow
            )


            # -----------------------------------------------
            # Neighbor table
            # -----------------------------------------------

            wd = (
                1.0
                /
                (
                    d[
                        use
                    ]
                    +
                    EPS
                )
            )

            wd = (
                wd
                /
                wd.sum()
            )


            for rank, (
                idx_ref,
                weight
            ) in enumerate(
                zip(
                    use,
                    wd
                ),
                start=1
            ):

                rr = ref.iloc[
                    idx_ref
                ]

                neighbor_rows.append({
                    "patient_id":
                        pid,

                    "method":
                        method,

                    "k":
                        k,

                    "slope_weight":
                        sw,

                    "rank":
                        rank,

                    "ref_id":
                        int(
                            rr[
                                "ref_id"
                            ]
                        ),

                    "distance":
                        float(
                            d[
                                idx_ref
                            ]
                        ),

                    "weight":
                        float(
                            weight
                        ),

                    "ref_tmin":
                        float(
                            rr[
                                "t_min_true"
                            ]
                        ),

                    "ref_base_mut_rate":
                        rr.get(
                            "base_mut_rate",
                            np.nan
                        ),

                    "ref_dose_drug_1":
                        rr.get(
                            "dose_drug_1",
                            np.nan
                        ),

                    "ref_dose_drug_2":
                        rr.get(
                            "dose_drug_2",
                            np.nan
                        ),

                    "ref_turnover_rate":
                        rr.get(
                            "turnover_rate",
                            np.nan
                        ),

                    "ref_draw_m2":
                        rr.get(
                            "draw_m2",
                            np.nan
                        ),
                })


predictions = pd.DataFrame(
    patient_predictions
)

neighbors = pd.DataFrame(
    neighbor_rows
)


pred_path = (
    OUT /
    "abc_80patients_4scan_hybrid10k.csv"
)

neighbor_path = (
    OUT /
    "abc_80patients_4scan_hybrid10k_neighbors.csv"
)

predictions.to_csv(
    pred_path,
    index=False
)

neighbors.to_csv(
    neighbor_path,
    index=False
)


# ============================================================
# HELD-OUT TEST SUMMARY
# ============================================================

test_summary_rows = []


for method, g in predictions.groupby(
    "method"
):

    stats = summarize_predictions(
        g.rename(
            columns={
                "true_tmin":
                    "true_tmin",

                "pred_tmin_weighted":
                    "pred_tmin_weighted",
            }
        )
    )

    stats.update({
        "method":
            method,

        "k":
            int(
                g[
                    "k"
                ].iloc[0]
            ),

        "slope_weight":
            float(
                g[
                    "slope_weight"
                ].iloc[0]
            ),
    })

    test_summary_rows.append(
        stats
    )


test_summary = pd.DataFrame(
    test_summary_rows
)


test_summary.to_csv(
    OUT /
    "abc_80patients_4scan_hybrid10k_summary.csv",
    index=False
)


print()
print("=== HELD-OUT TEST PERFORMANCE ===")

print(
    test_summary.to_string(
        index=False
    )
)


# ============================================================
# PAIRED BASELINE VS SLOPE
# ============================================================

wide = predictions.pivot(
    index="patient_id",
    columns="method",
    values="abs_error_weighted"
)


if {
    "baseline_logN",
    "logN_plus_slope"
}.issubset(
    wide.columns
):

    b = wide[
        "baseline_logN"
    ]

    s = wide[
        "logN_plus_slope"
    ]

    print()
    print(
        "=== BASELINE VS SLOPE-AWARE ==="
    )

    print(
        "slope wins:",
        int(
            (s < b).sum()
        )
    )

    print(
        "ties:",
        int(
            np.isclose(
                s,
                b
            ).sum()
        )
    )

    print(
        "baseline wins:",
        int(
            (b < s).sum()
        )
    )

    print(
        "mean AE difference "
        "(slope - baseline):",
        float(
            np.mean(
                s - b
            )
        )
    )


print()
print("WROTE:")
print(cv_path)
print(cv_summary_path)
print(pred_path)
print(neighbor_path)
print(
    OUT /
    "abc_80patients_4scan_hybrid10k_summary.csv"
)

print()
print(
    "PASS: reference-only tuning "
    "and held-out 80-patient ABC complete."
)
