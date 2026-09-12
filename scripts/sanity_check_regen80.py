#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
from collections import Counter

import numpy as np
import pandas as pd


ROOT = Path(
    "/users/w/a/wang0207/cancer_rescue/cancer_sims"
)

RAW = (
    ROOT /
    "fixed80_hybrid_heterogeneous_regenerated_v1"
)

MANIFEST = (
    RAW /
    "manifest_regenerated80.csv"
)


def read_one(path):

    with path.open(
        newline=""
    ) as f:

        return next(
            csv.DictReader(f)
        )


def marker(path):
    return Path(
        str(path) + ".done"
    )


def check_manifest():

    m = pd.read_csv(
        MANIFEST
    )

    assert len(m) == 80
    assert m["patient_id"].nunique() == 80
    assert m["expansion_seed"].nunique() == 80
    assert m["d1_seed"].nunique() == 80

    assert m["base_mut_rate"].between(
        1e-8,
        5e-8
    ).all()

    assert m["dose_drug_1"].between(
        0.60,
        0.80
    ).all()

    assert m["turnover_rate"].between(
        0.10,
        0.30
    ).all()

    assert m["draw_m2"].between(
        0.25,
        0.65
    ).all()

    assert (
        m["dose_drug_2"] <
        m["dose_drug_1"]
    ).all()

    assert np.allclose(
        m["draw_m1"] +
        m["draw_m2"] +
        m["draw_m3"],
        1.0
    )

    print(
        "=== MANIFEST ==="
    )

    print(
        "patients:",
        len(m)
    )

    for c in [
        "base_mut_rate",
        "dose_drug_1",
        "dose_drug_2",
        "turnover_rate",
        "draw_m2",
    ]:
        print(
            f"{c:18s}",
            "min=",
            m[c].min(),
            "max=",
            m[c].max()
        )

    print(
        "PASS: manifest ranges and identities."
    )

    return m


def check_patient(
    pid,
    lags,
    reps
):

    pdir = (
        RAW /
        "patients" /
        f"patient_{pid:03d}"
    )

    snap = (
        pdir /
        "pretreatment.bin"
    )

    meta = (
        pdir /
        "pretreatment_meta.csv"
    )

    prefix_path = (
        pdir /
        "drug1_prefix.csv"
    )


    assert (
        snap.exists()
        and
        snap.stat().st_size > 0
    )

    assert marker(
        snap
    ).exists()

    assert meta.exists()

    sm = pd.read_csv(
        meta
    )

    assert len(sm) == 1

    assert (
        sm["status"].iloc[0] ==
        "viable"
    )


    assert (
        prefix_path.exists()
        and
        prefix_path.stat().st_size > 0
    )

    assert marker(
        prefix_path
    ).exists()


    prefix = pd.read_csv(
        prefix_path
    )

    prefix["t_treatment"] = (
        pd.to_numeric(
            prefix["t_treatment"],
            errors="raise"
        )
    )

    assert (
        prefix[
            "t_treatment"
        ].duplicated().sum()
        == 0
    )


    print()
    print(
        f"=== PATIENT {pid:03d} ==="
    )

    print(
        "snapshot N:",
        int(
            sm["N"].iloc[0]
        )
    )

    print(
        "prefix rows:",
        len(prefix)
    )

    print(
        "prefix t range:",
        prefix[
            "t_treatment"
        ].min(),
        "to",
        prefix[
            "t_treatment"
        ].max()
    )


    if len(prefix):

        n = pd.to_numeric(
            prefix["N"],
            errors="coerce"
        ).to_numpy()

        t = pd.to_numeric(
            prefix["t_treatment"],
            errors="coerce"
        ).to_numpy()

        tmin = int(
            t[
                np.argmin(n)
            ]
        )

        print(
            "prefix tmin:",
            tmin
        )

        print(
            "N start/min/end:",
            int(n[0]),
            int(n.min()),
            int(n[-1])
        )


    # --------------------------------------------------------
    # Counterfactual integrity.
    #
    # For lag l, the state stored at the end of D1 cycle l-1
    # must agree with the one fixed prefix.
    # --------------------------------------------------------

    for lag in lags:

        row_prefix = prefix[
            prefix[
                "t_treatment"
            ].eq(
                lag - 1
            )
        ]


        outputs = []

        for rep in reps:

            f = (
                pdir /
                "outcomes" /
                f"lag_{lag:02d}" /
                f"rep_{rep:03d}.csv"
            )

            assert (
                f.exists()
                and
                f.stat().st_size > 0
            ), f

            assert marker(
                f
            ).exists()

            outputs.append(
                read_one(f)
            )


        seeds = [
            int(
                float(
                    x[
                        "treatment_seed"
                    ]
                )
            )
            for x in outputs
        ]

        assert (
            len(set(seeds)) ==
            len(seeds)
        )


        if len(row_prefix) == 1:

            expected_N = float(
                row_prefix[
                    "N"
                ].iloc[0]
            )

            expected_R2 = float(
                row_prefix[
                    "mean_R2_eff"
                ].iloc[0]
            )


            Ns = np.array([
                float(
                    x[
                        "N_pre_switch"
                    ]
                )
                for x in outputs
            ])

            R2s = np.array([
                float(
                    x[
                        "mean_R2_eff_pre_switch"
                    ]
                )
                for x in outputs
            ])


            assert np.allclose(
                Ns,
                expected_N,
                rtol=0,
                atol=1e-6
            ), (
                lag,
                Ns,
                expected_N
            )

            assert np.allclose(
                R2s,
                expected_R2,
                rtol=1e-5,
                atol=1e-8
            ), (
                lag,
                R2s,
                expected_R2
            )


            print(
                f"lag {lag:02d}: "
                f"fixed-prefix PASS | "
                f"N={expected_N:.0f} | "
                f"mean R2eff={expected_R2:.6g}"
            )

        else:

            assert all(
                int(
                    float(
                        x["switched"]
                    )
                ) == 0
                for x in outputs
            )

            print(
                f"lag {lag:02d}: "
                "D1 trajectory terminated before switch."
            )


def full_check(
    m,
    n_reps
):

    expected_outcomes = (
        80 *
        20 *
        n_reps
    )

    snapshot_done = list(
        RAW.glob(
            "patients/patient_*/pretreatment.bin.done"
        )
    )

    prefix_done = list(
        RAW.glob(
            "patients/patient_*/drug1_prefix.csv.done"
        )
    )

    outcome_done = list(
        RAW.glob(
            "patients/patient_*/outcomes/lag_*/rep_*.csv.done"
        )
    )


    print()
    print(
        "=== FULL COMPLETION ==="
    )

    print(
        "snapshots:",
        len(snapshot_done),
        "/ 80"
    )

    print(
        "prefixes:",
        len(prefix_done),
        "/ 80"
    )

    print(
        "outcomes:",
        len(outcome_done),
        f"/ {expected_outcomes}"
    )


    assert len(snapshot_done) == 80
    assert len(prefix_done) == 80
    assert len(outcome_done) == expected_outcomes


    statuses = Counter()

    unresolved = 0
    switched = 0

    for marker_path in outcome_done:

        f = Path(
            str(marker_path)[
                :-len(".done")
            ]
        )

        x = read_one(
            f
        )

        statuses[
            x["status"]
        ] += 1

        unresolved += int(
            x["status"] ==
            "unresolved"
        )

        switched += int(
            float(
                x["switched"]
            )
        )


    print()
    print(
        "statuses:"
    )

    for k, v in sorted(
        statuses.items()
    ):
        print(
            f"  {k:22s} {v}"
        )


    print(
        "fraction reaching D2:",
        switched /
        expected_outcomes
    )

    print(
        "unresolved fraction:",
        unresolved /
        expected_outcomes
    )


    # --------------------------------------------------------
    # Strong integrity check on EVERY patient and lag.
    # We only need reps 1 and n_reps here to prove that the
    # fixed D1 branch is identical across the replicate range.
    # --------------------------------------------------------

    for pid in range(80):

        check_patient(
            pid,
            range(1,21),
            sorted(
                set(
                    [
                        1,
                        n_reps
                    ]
                )
            )
        )


    errors = [
        p
        for p in (
            RAW /
            "errors"
        ).glob("*.err")
        if p.stat().st_size > 0
    ]

    print()
    print(
        "nonempty error files:",
        len(errors)
    )

    assert not errors

    print()
    print(
        "PASS: regenerated cohort is complete and "
        "counterfactual D1 trajectories are fixed."
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=[
            "smoke",
            "full"
        ],
        required=True
    )

    ap.add_argument(
        "--n-reps",
        type=int,
        default=10
    )

    args = ap.parse_args()


    m = check_manifest()


    if args.mode == "smoke":

        check_patient(
            0,
            [
                5,
                15
            ],
            [
                1,
                2
            ]
        )

        print()
        print(
            "PASS: SMOKE TEST READY FOR FULL RUN."
        )

    else:

        full_check(
            m,
            args.n_reps
        )


if __name__ == "__main__":
    main()
