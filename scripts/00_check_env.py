"""Stage 0 — environment self-check before running the pipeline (read-only; no patient-level data is read).

Checks, each printed as OK / FAIL (WARN = informational, does not fail):
  1. Python and the key packages match the versions pinned in environment.yml;
  2. MIMIC_ROOT (environment variable, else paths.mimic_root in config/config.yaml) exists and contains every raw
     hosp/icu table the pipeline reads (file existence only — no file is opened);
  3. external/mimic-code exists and its checked-out commit equals external/MIMIC_CODE_COMMIT.txt;
  4. physical memory (≥ 8 GB) and free disk space in the project folder (≥ 10 GB); available memory as WARN.
Exit code 0 if nothing failed, 1 otherwise (run_all.sh / run_stage8.sh stop on failure).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []


def report(status: str, item: str, detail: str = "") -> None:
    results.append((status, item, detail))
    print(f"[{status:4s}] {item}" + (f" — {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------- 1. Python and package versions
pins = dict(re.findall(r"^\s*-\s*([A-Za-z0-9_.\-]+)=([^=\s]+)", (ROOT / "environment.yml").read_text(), flags=re.M))
py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
report("OK" if py == pins.get("python") else "FAIL", "Python", f"{py} (environment.yml: {pins.get('python')})")
PACKAGES = {  # conda name in environment.yml -> installed distribution name
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "scikit-learn", "xgboost": "xgboost",
    "shap": "shap", "statsmodels": "statsmodels", "duckdb": "duckdb", "pyarrow": "pyarrow", "matplotlib": "matplotlib",
    "pyyaml": "PyYAML", "joblib": "joblib"}
for conda_name, dist in PACKAGES.items():
    want = pins.get(conda_name)
    try:
        have = metadata.version(dist)
    except metadata.PackageNotFoundError:
        report("FAIL", f"package {conda_name}", f"not installed (environment.yml: {want})")
        continue
    report("OK" if have == want else "FAIL", f"package {conda_name}", f"{have} (environment.yml: {want})")

# ---------------------------------------------------------------- 2. MIMIC-IV raw files (existence only)
try:
    from src.common import path, raw_file
    from src.subset_db import SUBSET_TABLES
    root = path("mimic_root")
    report("OK", "MIMIC_ROOT", str(root))
    missing = []
    for module, table, _ in SUBSET_TABLES:
        try:
            raw_file(module, table)
        except FileNotFoundError:
            missing.append(f"{module}/{table}")
    report("OK" if not missing else "FAIL", f"MIMIC-IV tables ({len(SUBSET_TABLES)} hosp/icu files, .csv.gz or .csv)",
           "all present" if not missing else "missing: " + ", ".join(missing))
except FileNotFoundError as e:
    report("FAIL", "MIMIC_ROOT", str(e).replace("\n", " "))
except Exception as e:  # e.g. a package import failed above
    report("FAIL", "MIMIC_ROOT", f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 3. mimic-code clone and commit
mc = ROOT / "external" / "mimic-code"
want_commit = (ROOT / "external" / "MIMIC_CODE_COMMIT.txt").read_text().split()[0]
if not (mc / "mimic-iv" / "concepts_duckdb").is_dir():
    report("FAIL", "external/mimic-code", f"not found; clone it (see REPRODUCE.md) and check out {want_commit}")
else:
    r = subprocess.run(["git", "-C", str(mc), "rev-parse", "HEAD"], capture_output=True, text=True)
    have_commit = r.stdout.strip()
    report("OK" if have_commit == want_commit else "FAIL", "external/mimic-code commit",
           f"{have_commit or r.stderr.strip()} (expected {want_commit})")

# ---------------------------------------------------------------- 4. memory and disk
def sysctl(name: str) -> int:
    return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip())

try:
    total = sysctl("hw.memsize")
    report("OK" if total >= 8 * 1024 ** 3 else "FAIL", "physical memory", f"{total / 1024 ** 3:.1f} GB (need ≥ 8 GB; DuckDB limit 4 GB)")
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", vm).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)\.", vm)}
    avail = page * sum(pages.values())
    report("OK" if avail >= 4 * 1024 ** 3 else "WARN", "available memory now",
           f"{avail / 1024 ** 3:.1f} GB (free + inactive + speculative; close other apps if < 4 GB)")
except Exception as e:
    report("WARN", "memory", f"could not determine ({type(e).__name__}: {e}); macOS sysctl/vm_stat expected")
free = shutil.disk_usage(ROOT).free
report("OK" if free >= 10 * 1024 ** 3 else "FAIL", "free disk space (project folder)",
       f"{free / 1024 ** 3:.0f} GB (need ≥ 10 GB for data/ and DuckDB temporary files)")

n_fail = sum(s == "FAIL" for s, _, _ in results)
print(f"\n{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'} "
      f"({sum(s == 'OK' for s, _, _ in results)} OK, {sum(s == 'WARN' for s, _, _ in results)} WARN, {n_fail} FAIL)")
sys.exit(1 if n_fail else 0)
