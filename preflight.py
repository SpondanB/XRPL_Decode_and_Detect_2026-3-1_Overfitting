"""Check the Python environment and supplied dataset before starting work."""

import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_PACKAGES = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("scikit-learn", "sklearn"),
    ("jupyter", "jupyter"),
)
REQUIRED_FILES = (
    "payments.py",
    "train/blocks.parquet",
    "train/labels.csv",
    "train/market_rates.csv",
    "train/transactions.parquet",
    "train/transfers.parquet",
    "validation/blocks.parquet",
    "validation/market_rates.csv",
    "validation/transactions.parquet",
    "validation/transfers.parquet",
)

results = []


def report(name, passed, detail):
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


def check_environment():
    in_venv = sys.prefix != sys.base_prefix
    report(
        "Virtual environment",
        in_venv,
        f"{sys.executable} ({'active' if in_venv else 'not active'})",
    )
    report("Python", sys.version_info >= (3, 10), sys.version.split()[0])

    requirements_path = ROOT / "requirements.txt"
    declared = set()
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip()
            declared.add(name.lower().replace("_", "-"))

    undeclared = [
        package
        for package, _ in REQUIRED_PACKAGES
        if package.lower().replace("_", "-") not in declared
    ]
    report(
        "Dependency manifest",
        not undeclared,
        "all required packages are declared"
        if not undeclared
        else f"missing declarations: {', '.join(undeclared)}",
    )

    missing = []
    installed = []
    for package, module in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
        except ImportError as error:
            missing.append(f"{package} ({error})")
        else:
            installed.append(f"{package} {importlib.metadata.version(package)}")
    report(
        "Required packages",
        not missing,
        ", ".join(installed + ([f"missing or broken: {', '.join(missing)}"] if missing else [])),
    )

    if in_venv:
        jupyter = Path(sys.executable).with_name("jupyter")
        if not jupyter.is_file():
            report("Jupyter command", False, f"not found beside {sys.executable}")
        else:
            jupyter_check = subprocess.run(
                [str(jupyter), "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            report(
                "Jupyter command",
                jupyter_check.returncode == 0,
                jupyter_check.stdout.splitlines()[0]
                if jupyter_check.returncode == 0 and jupyter_check.stdout
                else jupyter_check.stderr.strip() or "command failed",
            )
    else:
        print("[SKIP] Jupyter command: activate the project virtual environment first")

    if in_venv:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
        report(
            "Package consistency",
            pip_check.returncode == 0,
            pip_check.stdout.strip()
            or pip_check.stderr.strip()
            or "pip check found no conflicts",
        )
    else:
        print("[SKIP] Package consistency: activate the project virtual environment first")


def check_files_and_checksums():
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]
    report(
        "Required data files",
        not missing,
        "training and validation files are present"
        if not missing
        else f"missing: {', '.join(missing)}",
    )

    manifest_path = ROOT / "checksums.json"
    try:
        checksums = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        report("Checksum manifest", False, str(error))
        return

    mismatches = []
    absent = []
    for relative, expected in checksums.items():
        path = ROOT / relative
        if not path.is_file():
            absent.append(relative)
            continue
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected:
            mismatches.append(relative)

    okay = not mismatches and not absent
    detail = f"{len(checksums)} files match checksums"
    if absent:
        detail += f"; missing: {', '.join(absent)}"
    if mismatches:
        detail += f"; mismatched: {', '.join(mismatches)}"
    report("File checksums", okay, detail)


def read_csv_rows(relative, required_columns):
    with (ROOT / relative).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required_columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{relative} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def check_csv_data():
    labels = read_csv_rows("train/labels.csv", ("TRANSACTION_HASH", "is_suspicious"))
    label_hashes = [row["TRANSACTION_HASH"] for row in labels]
    label_values = {row["is_suspicious"] for row in labels}
    valid_labels = (
        len(label_hashes) == len(set(label_hashes))
        and label_values <= {"0", "1"}
        and bool(labels)
    )
    report(
        "Training labels",
        valid_labels,
        f"{len(labels)} rows, unique transaction hashes, binary labels"
        if valid_labels
        else "labels must be non-empty, unique by transaction hash, and 0 or 1",
    )

    for split in ("train", "validation"):
        rates = read_csv_rows(
            f"{split}/market_rates.csv", ("TIMESTAMP", "ASSET", "PRICE_MICRO_USD")
        )
        keys = [(row["TIMESTAMP"], row["ASSET"]) for row in rates]
        valid_rates = (
            bool(rates)
            and len(keys) == len(set(keys))
            and all(int(row["PRICE_MICRO_USD"]) > 0 for row in rates)
        )
        report(
            f"{split.title()} market rates",
            valid_rates,
            f"{len(rates)} rows with unique timestamp/asset pairs and positive prices"
            if valid_rates
            else "rates must be non-empty, unique by timestamp/asset, and positive",
        )


def check_notebook():
    notebook_path = ROOT / "starter_notebook.ipynb"
    try:
        notebook = json.loads(notebook_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        report("Starter notebook", False, str(error))
        return

    code_cells = [
        (index, cell)
        for index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code"
    ]
    syntax_errors = []
    for index, cell in code_cells:
        try:
            ast.parse("".join(cell.get("source", [])))
        except SyntaxError as error:
            syntax_errors.append(f"cell {index}: {error.msg} at line {error.lineno}")
    report(
        "Starter notebook syntax",
        bool(code_cells) and not syntax_errors,
        f"{len(code_cells)} code cells parse"
        if code_cells and not syntax_errors
        else "; ".join(syntax_errors) or "no code cells found",
    )


def check_parquet_data():
    if importlib.util.find_spec("pyarrow") is None:
        print("[SKIP] Parquet data: install requirements to inspect Parquet tables")
        return

    import pyarrow.parquet as parquet

    transaction_hashes = {}
    for split in ("train", "validation"):
        for table_name in ("blocks", "transactions", "transfers"):
            relative = f"{split}/{table_name}.parquet"
            path = ROOT / relative
            if not path.is_file():
                continue
            table = parquet.read_table(path)
            report(
                f"{relative} readability",
                table.num_rows > 0,
                f"{table.num_rows} rows; {table.num_columns} columns",
            )
            if table_name == "transactions":
                required = {
                    "TRANSACTION_HASH",
                    "TIMESTAMP",
                    "ACCOUNT",
                    "EXTRA_TRANSACTION_FIELDS",
                    "DELIVERED_AMOUNT",
                }
                missing = required - set(table.column_names)
                hashes = table.column("TRANSACTION_HASH").to_pylist()
                unique = len(hashes) == len(set(hashes))
                report(
                    f"{split.title()} transaction schema",
                    not missing and unique,
                    "required payment fields present; transaction hashes are unique"
                    if not missing and unique
                    else f"missing fields: {', '.join(sorted(missing))}; "
                    "transaction hashes must be unique",
                )
                transaction_hashes[split] = set(hashes)

    if "train" in transaction_hashes:
        labels = read_csv_rows("train/labels.csv", ("TRANSACTION_HASH",))
        label_hashes = {row["TRANSACTION_HASH"] for row in labels}
        report(
            "Training label alignment",
            label_hashes == transaction_hashes["train"],
            f"{len(label_hashes)} labels and "
            f"{len(transaction_hashes['train'])} transactions"
            if label_hashes == transaction_hashes["train"]
            else "label and transaction hashes do not match exactly",
        )

    if "train" in transaction_hashes and "validation" in transaction_hashes:
        overlap = transaction_hashes["train"] & transaction_hashes["validation"]
        report(
            "Train/validation separation",
            not overlap,
            "transaction hashes are disjoint"
            if not overlap
            else f"{len(overlap)} transaction hashes overlap",
        )

    from payments import load_payments

    for split in ("train", "validation"):
        if split not in transaction_hashes:
            continue
        payments = load_payments(ROOT / split)
        report(
            f"{split.title()} payment loader",
            len(payments) == len(transaction_hashes[split]),
            f"loaded {len(payments)} payments",
        )


def main():
    check_environment()
    check_files_and_checksums()
    csv_files = (
        ROOT / "train/labels.csv",
        ROOT / "train/market_rates.csv",
        ROOT / "validation/market_rates.csv",
    )
    if all(path.is_file() for path in csv_files):
        check_csv_data()
    else:
        print("[SKIP] CSV data: required CSV files are missing")
    if (ROOT / "starter_notebook.ipynb").is_file():
        check_notebook()
    else:
        print("[SKIP] Starter notebook: file is missing")
    if importlib.util.find_spec("pyarrow") is not None:
        check_parquet_data()
    else:
        print("[SKIP] Parquet data: install requirements to inspect Parquet tables")

    failures = sum(not passed for passed in results)
    print(f"\nPre-coding check: {'PASS' if failures == 0 else f'{failures} check(s) failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
