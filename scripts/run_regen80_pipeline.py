#!/usr/bin/env python3

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import os
import subprocess
import time

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

SLIM = Path(
    "/nas/longleaf/rhel9/apps/slim/5.1/bin/slim"
)

MODEL_EXPAND = (
    ROOT /
    "sims" /
    "hybrid_regen80_expand.slim"
)

MODEL_PREFIX = (
    ROOT /
    "sims" /
    "hybrid_regen80_prefix.slim"
)

MODEL_OUTCOME = (
    ROOT /
    "sims" /
    "hybrid_regen80_outcome.slim"
)


def patient_dir(pid):
    return (
        RAW /
        "patients" /
        f"patient_{int(pid):03d}"
    )


def q(s):
    # Eidos single-quoted string literal.
    return "'" + str(s).replace("'", "\\'") + "'"


def darg(cmd, key, value, string=False):

    if string:
        expr = q(value)
    elif isinstance(value, int):
        expr = str(value)
    else:
        expr = repr(float(value))

    cmd.extend(
        [
            "-d",
            f"{key}={expr}"
        ]
    )


def marker(path):
    return Path(
        str(path) + ".done"
    )


def is_done(path):
    return (
        path.exists()
        and
        path.stat().st_size > 0
        and
        marker(path).exists()
    )


def write_done(path, text="OK"):
    marker(path).write_text(
        text + "\n"
    )


def run_slim(
    cmd,
    error_path
):

    p = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if p.returncode != 0:

        error_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        error_path.write_text(
            p.stderr or
            f"SLiM exited {p.returncode}\n"
        )

        raise RuntimeError(
            f"SLiM failed: {error_path}"
        )

    if error_path.exists():
        error_path.unlink()


def base_defs(
    cmd,
    row
):

    darg(
        cmd,
        "patient_id",
        int(row["patient_id"])
    )

    darg(
        cmd,
        "base_mut_rate",
        row["base_mut_rate"]
    )

    darg(
        cmd,
        "dose_drug_1_mono",
        row["dose_drug_1"]
    )

    darg(
        cmd,
        "dose_drug_2_mono",
        row["dose_drug_2"]
    )

    darg(
        cmd,
        "patient_turnover",
        row["turnover_rate"]
    )

    darg(
        cmd,
        "draw_m1",
        row["draw_m1"]
    )

    darg(
        cmd,
        "draw_m2",
        row["draw_m2"]
    )

    darg(
        cmd,
        "draw_m3",
        row["draw_m3"]
    )

    darg(
        cmd,
        "cross_resistance",
        row["cross_resistance"]
    )


def read_one_csv(path):

    with path.open(
        newline=""
    ) as f:

        return next(
            csv.DictReader(f)
        )


# ============================================================
# PHASE A
# ============================================================

def make_snapshot(row):

    row = dict(row)

    pid = int(
        row["patient_id"]
    )

    pdir = patient_dir(pid)

    pdir.mkdir(
        parents=True,
        exist_ok=True
    )

    snap = (
        pdir /
        "pretreatment.bin"
    )

    meta = (
        pdir /
        "pretreatment_meta.csv"
    )

    if is_done(snap):
        return (
            pid,
            "snapshot_skip"
        )


    base_seed = int(
        row["expansion_seed"]
    )


    # Rare pre-treatment extinction is biological.
    # Keep parameters fixed and only retry the expansion RNG.
    for attempt in range(20):

        seed = (
            base_seed +
            attempt * 10000
        )

        snap.unlink(
            missing_ok=True
        )

        meta.unlink(
            missing_ok=True
        )

        cmd = [
            str(SLIM),
            "-s",
            str(seed),
        ]

        base_defs(
            cmd,
            row
        )

        darg(
            cmd,
            "expansion_seed",
            seed
        )

        darg(
            cmd,
            "snapshot_file",
            snap,
            string=True
        )

        darg(
            cmd,
            "meta_file",
            meta,
            string=True
        )

        cmd.append(
            str(MODEL_EXPAND)
        )


        err = (
            RAW /
            "errors" /
            f"snapshot_patient_{pid:03d}.err"
        )

        run_slim(
            cmd,
            err
        )


        if not meta.exists():
            raise RuntimeError(
                f"No metadata generated for patient {pid}"
            )


        info = read_one_csv(
            meta
        )


        if (
            info["status"] ==
            "viable"
        ):
            if (
                not snap.exists()
                or
                snap.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"Viable patient {pid} has no snapshot"
                )

            write_done(
                snap,
                f"resolved_expansion_seed={seed}"
            )

            return (
                pid,
                f"snapshot_ok_attempt_{attempt}"
            )


    raise RuntimeError(
        f"Patient {pid}: 20 expansion attempts all extinct."
    )


# ============================================================
# PHASE B
# ============================================================

def make_prefix(row):

    row = dict(row)

    pid = int(
        row["patient_id"]
    )

    pdir = patient_dir(pid)

    snap = (
        pdir /
        "pretreatment.bin"
    )

    prefix = (
        pdir /
        "drug1_prefix.csv"
    )


    if is_done(prefix):
        return (
            pid,
            "prefix_skip"
        )


    if not is_done(snap):
        raise RuntimeError(
            f"Snapshot not ready: patient {pid}"
        )


    prefix.unlink(
        missing_ok=True
    )


    cmd = [
        str(SLIM)
    ]

    base_defs(
        cmd,
        row
    )

    darg(
        cmd,
        "d1_seed",
        int(
            row["d1_seed"]
        )
    )

    darg(
        cmd,
        "snapshot_file",
        snap,
        string=True
    )

    darg(
        cmd,
        "prefix_file",
        prefix,
        string=True
    )

    cmd.append(
        str(MODEL_PREFIX)
    )


    err = (
        RAW /
        "errors" /
        f"prefix_patient_{pid:03d}.err"
    )


    run_slim(
        cmd,
        err
    )


    if (
        not prefix.exists()
        or
        prefix.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Missing prefix patient {pid}"
        )


    write_done(
        prefix
    )


    return (
        pid,
        "prefix_ok"
    )


# ============================================================
# PHASE C
# One worker receives one patient × lag and runs all replicate
# branches sequentially.
# ============================================================

def outcome_batch(args):

    row, lag, reps = args

    row = dict(row)

    pid = int(
        row["patient_id"]
    )

    pdir = patient_dir(pid)

    snap = (
        pdir /
        "pretreatment.bin"
    )

    odir = (
        pdir /
        "outcomes" /
        f"lag_{int(lag):02d}"
    )

    odir.mkdir(
        parents=True,
        exist_ok=True
    )


    completed = 0
    skipped = 0


    for rep in reps:

        outfile = (
            odir /
            f"rep_{int(rep):03d}.csv"
        )


        if is_done(outfile):

            skipped += 1
            continue


        outfile.unlink(
            missing_ok=True
        )


        # < signed 32-bit max.
        treatment_seed = (
            1_000_000_000
            +
            pid * 100_000
            +
            int(lag) * 100
            +
            int(rep)
        )


        cmd = [
            str(SLIM)
        ]

        base_defs(
            cmd,
            row
        )


        darg(
            cmd,
            "d1_seed",
            int(
                row["d1_seed"]
            )
        )

        darg(
            cmd,
            "second_strike_lag",
            int(lag)
        )

        darg(
            cmd,
            "replicate_id",
            int(rep)
        )

        darg(
            cmd,
            "treatment_seed",
            treatment_seed
        )

        darg(
            cmd,
            "snapshot_file",
            snap,
            string=True
        )

        darg(
            cmd,
            "outcome_file",
            outfile,
            string=True
        )


        cmd.append(
            str(MODEL_OUTCOME)
        )


        err = (
            RAW /
            "errors" /
            (
                f"outcome_p{pid:03d}"
                f"_lag{int(lag):02d}"
                f"_rep{int(rep):03d}.err"
            )
        )


        run_slim(
            cmd,
            err
        )


        if (
            not outfile.exists()
            or
            outfile.stat().st_size == 0
        ):
            raise RuntimeError(
                f"Missing outcome {outfile}"
            )


        write_done(
            outfile
        )

        completed += 1


    return (
        pid,
        lag,
        completed,
        skipped
    )


def run_parallel(
    fn,
    items,
    workers,
    name
):

    print(
        f"\n=== {name}: "
        f"{len(items)} work units, "
        f"{workers} workers ===",
        flush=True
    )

    done = 0

    with ProcessPoolExecutor(
        max_workers=workers
    ) as ex:

        futures = [
            ex.submit(
                fn,
                x
            )
            for x in items
        ]


        for fut in as_completed(
            futures
        ):

            result = fut.result()

            done += 1

            if (
                done <= 5
                or
                done % 25 == 0
                or
                done == len(items)
            ):
                print(
                    f"{name}: "
                    f"{done}/{len(items)} "
                    f"{result}",
                    flush=True
                )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK",
                "2"
            )
        )
    )

    ap.add_argument(
        "--n-reps",
        type=int,
        default=10
    )

    ap.add_argument(
        "--sanity",
        action="store_true"
    )

    args = ap.parse_args()


    if not MANIFEST.exists():
        raise SystemExit(
            f"Missing manifest: {MANIFEST}"
        )


    df = pd.read_csv(
        MANIFEST
    )


    if args.sanity:

        df = df[
            df["patient_id"].eq(0)
        ].copy()

        lags = [
            5,
            15
        ]

        reps = [
            1,
            2
        ]

        workers = min(
            args.workers,
            2
        )

    else:

        lags = list(
            range(1, 21)
        )

        reps = list(
            range(
                1,
                args.n_reps + 1
            )
        )

        workers = args.workers


    rows = (
        df.to_dict(
            "records"
        )
    )


    t0 = time.time()


    run_parallel(
        make_snapshot,
        rows,
        workers,
        "SNAPSHOTS"
    )


    run_parallel(
        make_prefix,
        rows,
        workers,
        "D1 PREFIXES"
    )


    batches = [
        (
            row,
            lag,
            reps
        )
        for row in rows
        for lag in lags
    ]


    run_parallel(
        outcome_batch,
        batches,
        workers,
        "TWO-STRIKE OUTCOMES"
    )


    elapsed = (
        time.time() -
        t0
    ) / 3600.0


    print(
        f"\nPIPELINE COMPLETE "
        f"in {elapsed:.2f} hours.",
        flush=True
    )


if __name__ == "__main__":
    main()
