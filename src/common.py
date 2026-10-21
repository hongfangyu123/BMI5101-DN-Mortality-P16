"""Shared helpers: configuration, DuckDB connections, typed access to raw MIMIC-IV files.

Compliance: nothing in this module prints patient-level rows. Callers must only
emit aggregates (counts, rates, summary statistics).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def mimic_root() -> Path:
    """MIMIC-IV v3.1 root (the folder containing hosp/ and icu/): environment variable MIMIC_ROOT if set, otherwise
    paths.mimic_root in config/config.yaml (`~` expanded). Raises with setup instructions if the folder does not exist."""
    env = os.environ.get("MIMIC_ROOT")
    src = "environment variable MIMIC_ROOT" if env else "paths.mimic_root in config/config.yaml"
    p = Path(env or load_config()["paths"]["mimic_root"]).expanduser()
    p = p if p.is_absolute() else ROOT / p
    if not p.is_dir():
        raise FileNotFoundError(
            f"MIMIC-IV v3.1 folder not found: {p} (taken from {src}).\n"
            "Point it at the folder that contains hosp/ and icu/, e.g.\n"
            "    export MIMIC_ROOT=/path/to/mimic-iv-3.1\n"
            "or edit paths.mimic_root in config/config.yaml. See REPRODUCE.md.")
    return p


def path(key: str) -> Path:
    """Resolve a path from config (relative paths are relative to the project root; `~` is expanded).
    mimic_root honours the MIMIC_ROOT environment variable (see mimic_root())."""
    if key == "mimic_root":
        return mimic_root()
    p = Path(load_config()["paths"][key]).expanduser()
    return p if p.is_absolute() else ROOT / p


def connect(database: str | Path = ":memory:", read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = load_config()["duckdb"]
    tmp = path("duckdb_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database), read_only=read_only)
    con.execute(f"SET memory_limit='{cfg['memory_limit']}'")
    con.execute(f"SET threads={int(cfg['threads'])}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def raw_file(module: str, table: str) -> Path:
    """Path of a raw MIMIC-IV table (module = 'hosp' or 'icu'); prefers .csv.gz."""
    base = path("mimic_root") / module
    for ext in (".csv.gz", ".csv"):
        f = base / f"{table}{ext}"
        if f.exists():
            return f
    raise FileNotFoundError(f"{module}/{table} not found under {base}")


def patched_create_sql() -> str:
    """mimic-code's Postgres schema, patched for DuckDB exactly as buildmimic/duckdb/build_mimic.sh does,
    plus FLOAT -> DOUBLE (DuckDB FLOAT is 4-byte; Postgres FLOAT is 8-byte)."""
    sql = (path("mimic_code") / "mimic-iv/buildmimic/postgres/create.sql").read_text()
    sql = re.sub(r"TIMESTAMP\(\d+\)", "TIMESTAMP", sql)
    sql = re.sub(r"(spec_type_desc.+)NOT NULL", r"\1", sql)
    sql = re.sub(r"(drug +VARCHAR.+)NOT NULL", r"\1", sql)
    sql = re.sub(r"\bFLOAT\b", "DOUBLE", sql)
    return sql


@lru_cache(maxsize=1)
def schema_columns() -> dict[str, list[tuple[str, str]]]:
    """{'mimiciv_hosp.admissions': [(col, duckdb_type), ...], ...} from the official schema."""
    con = duckdb.connect(":memory:")
    con.execute(patched_create_sql())
    rows = con.execute(
        "SELECT table_schema || '.' || table_name, column_name, data_type "
        "FROM information_schema.columns WHERE table_schema LIKE 'mimiciv_%' "
        "ORDER BY table_schema, table_name, ordinal_position"
    ).fetchall()
    con.close()
    out: dict[str, list[tuple[str, str]]] = {}
    for t, c, typ in rows:
        out.setdefault(t, []).append((c, typ))
    return out


def read_csv_expr(module: str, table: str) -> str:
    """SQL expression reading a raw file with the official column types (no type sniffing)."""
    cols = schema_columns()[f"mimiciv_{module}.{table}"]
    struct = ", ".join(f"'{c}': '{t}'" for c, t in cols)
    return (
        f"read_csv('{raw_file(module, table)}', header=true, delim=',', quote='\"', escape='\"', "
        f"columns={{{struct}}})"
    )


def create_raw_views(con: duckdb.DuckDBPyConnection, tables: list[tuple[str, str]], schema: str = "raw") -> None:
    """Views over raw files (no data copied)."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    for module, table in tables:
        con.execute(f"CREATE OR REPLACE VIEW {schema}.{table} AS SELECT * FROM {read_csv_expr(module, table)}")


def dn_codes() -> list[str]:
    c = load_config()["cohort"]
    return list(c["dn_icd9"]) + list(c["dn_icd10"])


def sql_list(values) -> str:
    return ", ".join(f"'{v}'" for v in values)
