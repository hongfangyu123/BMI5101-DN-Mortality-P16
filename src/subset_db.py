"""Build a cohort-only DuckDB database and derive mimic-code concepts in it.

The database (data/interim/mimic_subset.duckdb) holds the official mimiciv_hosp / mimiciv_icu
schemas, filled only with rows of the candidate patients, plus mimiciv_derived concept tables
computed by the unmodified mimic-code DuckDB SQL. Steps are recorded in a progress table so an
interrupted build resumes. Only aggregate row counts are logged.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from src.common import connect, path, patched_create_sql, read_csv_expr

# (module, table, filter by subject?)  — only the tables the needed concepts read
SUBSET_TABLES = [
    ("hosp", "d_labitems", False),
    ("icu", "d_items", False),
    ("hosp", "patients", True),
    ("hosp", "admissions", True),
    ("hosp", "diagnoses_icd", True),
    ("hosp", "services", True),
    ("icu", "icustays", True),
    ("icu", "outputevents", True),
    ("hosp", "labevents", True),
    ("icu", "chartevents", True),
]

# mimic-code concepts_duckdb files needed for the study, in dependency order (subset of duckdb.sql)
CONCEPTS = [
    "demographics/age.sql",
    "measurement/bg.sql",
    "measurement/blood_differential.sql",
    "measurement/chemistry.sql",
    "measurement/coagulation.sql",
    "measurement/complete_blood_count.sql",
    "measurement/enzyme.sql",
    "measurement/gcs.sql",
    "measurement/vitalsign.sql",
    "measurement/urine_output.sql",
    "measurement/oxygen_delivery.sql",
    "measurement/ventilator_setting.sql",
    "treatment/ventilation.sql",
    "comorbidity/charlson.sql",
    "firstday/first_day_bg.sql",
    "firstday/first_day_bg_art.sql",
    "firstday/first_day_gcs.sql",
    "firstday/first_day_lab.sql",
    "firstday/first_day_urine_output.sql",
    "firstday/first_day_vitalsign.sql",
    "score/apsiii.sql",
    "score/lods.sql",
    "score/oasis.sql",
    "score/sapsii.sql",
]

PROGRESS = "main.build_progress"


def _done(con, step: str) -> bool:
    return con.execute(f"SELECT count(*) FROM {PROGRESS} WHERE step = ?", [step]).fetchone()[0] > 0


def _mark(con, step: str) -> None:
    con.execute(f"INSERT INTO {PROGRESS} VALUES (?, now())", [step])


def build_subset_db(subject_ids: pd.Series, db_path: Path | None = None, log=print) -> Path:
    db_path = Path(db_path or path("subset_db"))
    ids = pd.DataFrame({"subject_id": pd.Series(subject_ids).drop_duplicates().astype("int64")})
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = connect(db_path)
    con.execute(f"CREATE TABLE IF NOT EXISTS {PROGRESS} (step VARCHAR PRIMARY KEY, done_at TIMESTAMP)")

    # A different subject set invalidates the database: rebuild from scratch.
    con.register("ids_df", ids)
    con.execute("CREATE TABLE IF NOT EXISTS main.subset_subjects AS SELECT subject_id FROM ids_df")
    same = con.execute(
        "SELECT (SELECT count(*) FROM main.subset_subjects) = (SELECT count(*) FROM ids_df) AND "
        "(SELECT count(*) FROM main.subset_subjects s SEMI JOIN ids_df d USING (subject_id)) = (SELECT count(*) FROM ids_df)"
    ).fetchone()[0]
    if not same:
        con.close()
        log("subject set changed -> rebuilding subset database")
        db_path.unlink()
        return build_subset_db(subject_ids, db_path, log)
    log(f"subset subjects: {len(ids)}")

    if not _done(con, "schema"):
        con.execute(patched_create_sql())
        _mark(con, "schema")

    for module, table, by_subject in SUBSET_TABLES:
        step = f"load:{module}.{table}"
        if _done(con, step):
            continue
        t0 = time.time()
        where = "WHERE subject_id IN (SELECT subject_id FROM main.subset_subjects)" if by_subject else ""
        con.execute(f"DELETE FROM mimiciv_{module}.{table}")
        con.execute(f"INSERT INTO mimiciv_{module}.{table} SELECT * FROM {read_csv_expr(module, table)} {where}")
        n = con.execute(f"SELECT count(*) FROM mimiciv_{module}.{table}").fetchone()[0]
        _mark(con, step)
        log(f"loaded mimiciv_{module}.{table}: {n:,} rows ({time.time() - t0:.0f}s)")

    concept_dir = path("mimic_code") / "mimic-iv/concepts_duckdb"
    for rel in CONCEPTS:
        step = f"concept:{rel}"
        if _done(con, step):
            continue
        t0 = time.time()
        con.execute((concept_dir / rel).read_text())
        _mark(con, step)
        name = Path(rel).stem
        n = con.execute(f"SELECT count(*) FROM mimiciv_derived.{name}").fetchone()[0]
        log(f"concept {name}: {n:,} rows ({time.time() - t0:.0f}s)")

    con.execute("CHECKPOINT")
    con.close()
    return db_path


def open_subset(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return connect(path("subset_db"), read_only=read_only)
