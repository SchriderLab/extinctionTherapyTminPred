#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/users/w/a/wang0207/cancer_rescue/cancer_sims"
)

OUT = (
    ROOT /
    "abc_ref_hybrid_regen80_v1_10k"
)

MANIFEST = (
    OUT /
    "manifest_abc_ref_hybrid_regen80_v1_10k.csv"
)

PATIENT_MANIFEST = (
    ROOT /
    "fixed80_hybrid_heterogeneous_regenerated_v1" /
    "manifest_regenerated80.csv"
)

SCAN_TIMES = [
    0,
    1,
    2,
    3,
]

MIN_USABLE_FRACTION = 0.90


def get_t_col(df):

    for c in [
        "t",
        "t_treatment",
        "treatment_t",
        "cycle",
    ]:

        if c in df.columns:
            return c

    return None


def trajectory_hash(x):

    payload = "\n".join(
        f"{float(t):.12g},{float(n):.12g}"
        for t, n in x[
            ["t", "N"]
        ].itertuples(
            index=False,
            name=None
        )
    )

    return hashlib.sha256(
        payload.encode()
    ).hexdigest()


def nearest_exact(
    traj,
    t
):

    z = traj[
        np.isclose(
            traj["t"],
            t,
            atol=1e-9
        )
    ]

    if len(z) == 0:
        return None

    return float(
        z.iloc[0]["N"]
    )


manifest = pd.read_csv(
    MANIFEST
)

assert len(manifest) == 10000


rows = []


for i, m in manifest.iterrows():

    rid = int(
        m["ref_id"]
    )

    prefix = (
        OUT /
        "prefixes" /
        f"ref_{rid:05d}.csv"
    )

    meta_file = (
        OUT /
        "meta" /
        f"ref_{rid:05d}.csv"
    )

    done = (
        OUT /
        "done" /
        f"ref_{rid:05d}.done"
    )

    rec = {
        "ref_id":
            rid,

        "prefix_exists":
            int(prefix.exists()),

        "done":
            int(done.exists()),

        "usable":
            0,

        "problem":
            "",
    }

    problems = []

    if (
        not prefix.exists()
        or
        prefix.stat().st_size == 0
    ):

        problems.append(
            "missing_prefix"
        )

        rec["problem"] = ";".join(
            problems
        )

        rows.append(rec)

        continue

    try:

        df = pd.read_csv(
            prefix
        )

    except Exception as exc:

        rec["problem"] = (
            f"unreadable_prefix:{exc}"
        )

        rows.append(rec)

        continue

    if "N" not in df.columns:

        rec["problem"] = (
            "missing_N"
        )

        rows.append(rec)

        continue

    tcol = get_t_col(
        df
    )

    if tcol is None:

        rec["problem"] = (
            "missing_time_column"
        )

        rows.append(rec)

        continue

    traj = pd.DataFrame({
        "t":
            pd.to_numeric(
                df[tcol],
                errors="coerce"
            ),

        "N":
            pd.to_numeric(
                df["N"],
                errors="coerce"
            ),
    })

    # Current prefix model normally writes t=0,... directly.
    # Fallback for absolute cycle numbering.
    traj = (
        traj
        .dropna()
        .sort_values("t")
        .drop_duplicates(
            "t",
            keep="last"
        )
    )

    if (
        tcol == "cycle"
        and
        traj["t"].min() > 3
    ):

        traj["t"] = (
            traj["t"]
            -
            traj["t"].min()
        )

    if len(traj) == 0:

        rec["problem"] = (
            "empty_trajectory"
        )

        rows.append(rec)

        continue

    # --------------------------------------------------------
    # True nadir
    # --------------------------------------------------------

    imin = traj[
        "N"
    ].idxmin()

    tmin = float(
        traj.loc[
            imin,
            "t"
        ]
    )

    Nmin = float(
        traj.loc[
            imin,
            "N"
        ]
    )

    max_t = float(
        traj["t"].max()
    )

    rec.update({
        "n_points":
            len(traj),

        "max_t":
            max_t,

        "t_min_true":
            tmin,

        "N_min":
            Nmin,

        "tmin_boundary":
            int(
                np.isclose(
                    tmin,
                    max_t
                )
            ),

        "trajectory_hash":
            trajectory_hash(
                traj
            ),
    })

    # --------------------------------------------------------
    # Exact four scans
    # --------------------------------------------------------

    scan_N = []

    for t in SCAN_TIMES:

        n = nearest_exact(
            traj,
            t
        )

        scan_N.append(
            n
        )

    if any(
        x is None
        for x in scan_N
    ):

        problems.append(
            "missing_scan_0_1_2_3"
        )

    # Same scientific criterion as old ABC:
    # all observations must precede tmin.
    if not (
        tmin >
        max(SCAN_TIMES)
    ):

        problems.append(
            "tmin_not_after_scan3"
        )

    if not problems:

        logN = [
            np.log10(
                n + 1.0
            )
            for n in scan_N
        ]

        rec.update({
            "N0":
                scan_N[0],

            "N1":
                scan_N[1],

            "N2":
                scan_N[2],

            "N3":
                scan_N[3],

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

            "usable":
                1,
        })

    # --------------------------------------------------------
    # Optional hidden-state summaries at tmin
    # --------------------------------------------------------

    for candidate in [
        "mean_R2_eff",
        "mean_R2",
        "freq_m2",
        "frac_R2_eff_ge_05",
        "frac_R2_eff_ge_09",
    ]:

        if candidate not in df.columns:
            continue

        z = pd.DataFrame({
            "t":
                pd.to_numeric(
                    df[tcol],
                    errors="coerce"
                ),

            "value":
                pd.to_numeric(
                    df[candidate],
                    errors="coerce"
                )
        }).dropna()

        if (
            tcol == "cycle"
            and
            len(z)
            and
            z["t"].min() > 3
        ):

            z["t"] -= (
                z["t"].min()
            )

        q = z[
            np.isclose(
                z["t"],
                tmin,
                atol=1e-9
            )
        ]

        if len(q):

            rec[
                candidate
                +
                "_at_tmin"
            ] = float(
                q.iloc[0]["value"]
            )

    # --------------------------------------------------------
    # Manifest parameters
    # --------------------------------------------------------

    for c in [
        "base_mut_rate",
        "dose_drug_1",
        "dose_drug_2",
        "turnover_rate",
        "draw_m1",
        "draw_m2",
        "draw_m3",
        "cross_resistance",
        "expansion_seed",
        "d1_seed",
    ]:

        rec[c] = m[c]

    if meta_file.exists():

        try:
            meta = pd.read_csv(
                meta_file
            )

            if len(meta):

                for c in [
                    "original_expansion_seed",
                    "resolved_expansion_seed",
                    "recovery_attempt",
                    "recovered",
                    "snapshot_N",
                ]:

                    if c in meta.columns:

                        rec[c] = (
                            meta.iloc[0][c]
                        )

        except Exception:
            pass

    rec["problem"] = ";".join(
        problems
    )

    rows.append(rec)

    if (
        (i + 1) % 500 == 0
        or
        i + 1 == len(manifest)
    ):

        print(
            f"checked {i + 1}/"
            f"{len(manifest)}",
            flush=True
        )


report = pd.DataFrame(
    rows
)

all_path = (
    OUT /
    "abc_reference_features_all.csv"
)

usable_path = (
    OUT /
    "abc_reference_features_usable.csv"
)

report.to_csv(
    all_path,
    index=False
)

usable = report[
    report["usable"] == 1
].copy()

usable.to_csv(
    usable_path,
    index=False
)


# ============================================================
# GLOBAL SANITY
# ============================================================

n_total = len(report)

n_done = int(
    report["done"].sum()
)

n_usable = len(
    usable
)

usable_fraction = (
    n_usable /
    n_total
)


print()
print("=" * 72)
print("HYBRID ABC REFERENCE LIBRARY SANITY")
print("=" * 72)

print(
    "manifest refs: ",
    n_total
)

print(
    "completed:     ",
    n_done
)

print(
    "ABC usable:    ",
    n_usable
)

print(
    "usable fraction:",
    f"{usable_fraction:.3%}"
)


print()
print("Problems:")

print(
    report[
        "problem"
    ]
    .replace(
        "",
        "OK"
    )
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Exact duplicate trajectories
# ------------------------------------------------------------

hashes = usable[
    "trajectory_hash"
].dropna()

n_dup = (
    len(hashes)
    -
    hashes.nunique()
)

dup_fraction = (
    n_dup /
    len(hashes)
    if len(hashes)
    else np.nan
)

print()
print(
    "unique usable hashes:",
    hashes.nunique()
)

print(
    "exact duplicates:",
    n_dup
)

print(
    "duplicate fraction:",
    dup_fraction
)


# ------------------------------------------------------------
# Recovery
# ------------------------------------------------------------

if (
    "recovered"
    in report.columns
):

    rr = pd.to_numeric(
        report["recovered"],
        errors="coerce"
    )

    print()
    print(
        "expansion seed recovery count:",
        int(
            (rr == 1).sum()
        )
    )


# ------------------------------------------------------------
# Parameter-distribution comparison against 80 patients
# ------------------------------------------------------------

if PATIENT_MANIFEST.exists():

    patients = pd.read_csv(
        PATIENT_MANIFEST
    )

    mapping = {
        "base_mut_rate":
            "base_mut_rate",

        "dose_drug_1":
            "dose_drug_1",

        "dose_drug_2":
            "dose_drug_2",

        "turnover_rate":
            "turnover_rate",

        "draw_m2":
            "draw_m2",
    }

    print()
    print(
        "=== PARAMETER COVERAGE: "
        "PATIENTS VS REFERENCE ==="
    )

    coverage_rows = []

    for rc, pc in mapping.items():

        r = pd.to_numeric(
            manifest[rc],
            errors="coerce"
        )

        p = pd.to_numeric(
            patients[pc],
            errors="coerce"
        )

        coverage_rows.append({
            "parameter":
                rc,

            "patient_min":
                p.min(),

            "patient_median":
                p.median(),

            "patient_max":
                p.max(),

            "reference_min":
                r.min(),

            "reference_median":
                r.median(),

            "reference_max":
                r.max(),
        })

    coverage = pd.DataFrame(
        coverage_rows
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    coverage.to_csv(
        OUT /
        "reference_vs_patient_parameter_coverage.csv",
        index=False
    )


# ------------------------------------------------------------
# Library summary JSON
# ------------------------------------------------------------

summary = {
    "attempted_references":
        int(n_total),

    "completed_references":
        int(n_done),

    "abc_usable_references":
        int(n_usable),

    "usable_fraction":
        float(usable_fraction),

    "unique_trajectory_hashes":
        int(
            hashes.nunique()
        ),

    "exact_duplicate_count":
        int(n_dup),

    "exact_duplicate_fraction":
        (
            float(dup_fraction)
            if np.isfinite(
                dup_fraction
            )
            else None
        ),

    "scan_times":
        SCAN_TIMES,

    "model":
        "hybrid quantitative resistance; "
        "same generator as regenerated80 cohort",
}

(
    OUT /
    "reference_library_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2
    )
)


# ============================================================
# HARD PASS/FAIL
# ============================================================

assert n_done == n_total, (
    f"Only {n_done}/{n_total} "
    "reference trajectories completed."
)

assert (
    usable_fraction
    >=
    MIN_USABLE_FRACTION
), (
    f"Only {usable_fraction:.3%} "
    "of references are ABC usable."
)

assert (
    dup_fraction
    < 0.50
), (
    "Catastrophic exact trajectory duplication."
)


print()
print(
    "WROTE:",
    all_path
)

print(
    "WROTE:",
    usable_path
)

print()
print(
    "PASS: hybrid ABC reference library "
    "is complete and internally consistent."
)
