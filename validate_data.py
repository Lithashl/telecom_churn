from pathlib import Path
import csv
import re
import sys
from collections import Counter

# Explicit data-quality validation, run against the raw CSV before any cleaning happens.
# Does NOT modify the CSV or touch the database — read-only report.
#
# Why this exists: pandas' .isnull() only catches real NaN, not "hidden" missing values
# like a lone whitespace character (" "). This dataset's TotalCharges column has exactly
# that trap. These checks catch it explicitly instead of assuming the data is clean.
DATA_FILE = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")

SUSPECT_TOKENS = {"", " ", "na", "n/a", "null", "none", "-", "?", "nan", "unknown"}
ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
CUSTOMER_ID_PATTERN = re.compile(r"^\d{4}-[A-Z]{5}$")


def load_rows():
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# Known, already-handled issue: TotalCharges has 11 rows containing a lone whitespace
# character. import_dataset.py strips it to "" -> NULL, and clean_data.py imputes +
# flags it. Any OTHER hidden-missing hit (different column, or a different token) is
# a genuine new finding and should fail the check.
KNOWN_HIDDEN_MISSING = {"TotalCharges": {"' '": 11}}


def check_hidden_missing(rows, fieldnames):
    hits = {}
    for col in fieldnames:
        counts = Counter()
        for row in rows:
            raw = row[col]
            if raw.strip().lower() in SUSPECT_TOKENS:
                counts[repr(raw)] += 1
        if counts:
            hits[col] = dict(counts)

    unexpected = {col: c for col, c in hits.items() if c != KNOWN_HIDDEN_MISSING.get(col)}
    ok = not unexpected
    if not hits:
        detail = "none found"
    elif ok:
        detail = f"{hits} (known, already handled by import_dataset.py/clean_data.py)"
    else:
        detail = f"unexpected: {unexpected}"
    return ok, f"Hidden-missing values (whitespace/'N/A'/'null'/etc, not caught by naive isnull()): {detail}"


def check_exact_duplicates(rows):
    seen = Counter(tuple(r.values()) for r in rows)
    dupes = sum(1 for v in seen.values() if v > 1)
    return dupes == 0, f"Exact duplicate rows (all columns identical): {dupes}"


def check_whitespace_padding(rows, fieldnames):
    padded = Counter()
    for row in rows:
        for col in fieldnames:
            val = row[col]
            if val != val.strip() and val.strip() != "":
                padded[col] += 1
    ok = not padded
    detail = "none" if ok else dict(padded)
    return ok, f"Leading/trailing whitespace on non-blank values: {detail}"


def check_phone_consistency(rows):
    mismatches = sum(
        1 for r in rows
        if (r["PhoneService"] == "No") != (r["MultipleLines"] == "No phone service")
    )
    return mismatches == 0, f"PhoneService vs MultipleLines logical consistency: {mismatches} mismatch(es)"


def check_internet_consistency(rows):
    mismatches = 0
    for r in rows:
        no_internet = r["InternetService"] == "No"
        for col in ADDON_COLUMNS:
            if (r[col] == "No internet service") != no_internet:
                mismatches += 1
                break
    return mismatches == 0, f"InternetService vs the 6 add-on columns logical consistency: {mismatches} mismatch(es)"


def check_tenure_totalcharges_crosscheck(rows):
    tenure0_not_blank = sum(1 for r in rows if r["tenure"].strip() == "0" and r["TotalCharges"].strip() != "")
    blank_not_tenure0 = sum(1 for r in rows if r["TotalCharges"].strip() == "" and r["tenure"].strip() != "0")
    ok = tenure0_not_blank == 0 and blank_not_tenure0 == 0
    return ok, (
        "tenure==0 <-> TotalCharges blank cross-check: "
        f"{tenure0_not_blank} tenure=0 row(s) with a non-blank TotalCharges, "
        f"{blank_not_tenure0} blank-TotalCharges row(s) with tenure!=0"
    )


def check_range_sanity(rows):
    issues = []
    tenures = [int(r["tenure"]) for r in rows]
    if min(tenures) < 0 or max(tenures) > 72:
        issues.append(f"tenure outside [0,72]: min={min(tenures)} max={max(tenures)}")
    monthly = [float(r["MonthlyCharges"]) for r in rows]
    if min(monthly) <= 0:
        issues.append(f"non-positive MonthlyCharges found: min={min(monthly)}")
    for r in rows:
        tc = r["TotalCharges"].strip()
        if tc == "":
            continue
        tc_val = float(tc)
        if tc_val < 0:
            issues.append(f"negative TotalCharges for {r['customerID']}")
        if int(r["tenure"]) >= 1 and tc_val < float(r["MonthlyCharges"]) - 0.01:
            issues.append(f"TotalCharges < MonthlyCharges despite tenure>=1 for {r['customerID']}")
    ok = not issues
    return ok, f"Range/sanity checks (tenure, charges): {'all within expected bounds' if ok else issues}"


def check_customer_id(rows):
    ids = [r["customerID"] for r in rows]
    dupes = len(ids) - len(set(ids))
    bad_format = [i for i in ids if not CUSTOMER_ID_PATTERN.match(i)]
    ok = dupes == 0 and not bad_format
    return ok, (
        f"customerID uniqueness & format (NNNN-AAAAA): {dupes} duplicate id(s), "
        f"{len(bad_format)} malformed id(s)"
    )


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run import_dataset.py first to download it.")

    rows = load_rows()
    fieldnames = list(rows[0].keys())

    checks = [
        check_hidden_missing(rows, fieldnames),
        check_exact_duplicates(rows),
        check_whitespace_padding(rows, fieldnames),
        check_phone_consistency(rows),
        check_internet_consistency(rows),
        check_tenure_totalcharges_crosscheck(rows),
        check_range_sanity(rows),
        check_customer_id(rows),
    ]

    print(f"Data validation report — {DATA_FILE} ({len(rows)} rows)")
    print("=" * 70)
    passed = 0
    for ok, message in checks:
        mark = "PASS" if ok else "FAIL"
        passed += ok
        print(f"[{mark}] {message}")
    print("=" * 70)
    print(f"{passed}/{len(checks)} checks passed.")

    if passed < len(checks):
        print("Unexpected data-quality issue(s) found above — review before trusting the data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
