"""
SCENTINEL - OBP panel selection (menu).

Choose the tool, answer the questions, and see the results using the real
SCENTINEL CSVs in data/:
    [1] Tool 1 - Specificity    : smallest OBP panel so nothing imitates the target VOC
    [2] Tool 2 - Discrimination : OBP panel that best separates healthy vs diseased
    [3] PDF report              : Tool 1 + Tool 2, both data modes, verdicts and lab priorities

Run: python main.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BINDING_FILE_EXPERIMENTAL = "Compound_OBP_binding.csv"
SEP = "=" * 82


def load_module(rel_path):
    """Import a script (tool or demo) from its path so its main() can be called."""
    path = os.path.join(SCRIPT_DIR, rel_path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose(title, options, default="1"):
    """Numbered menu in the SCENTINEL style; returns the chosen key."""
    print(f"\n{SEP}\n  {title}\n{SEP}")
    for key, text in options:
        print(f"  [{key}] {text}")
    keys = [k for k, _ in options]
    raw = input(f"\n  Choose [{'/'.join(keys)}, default {default}]: ").strip()
    return raw if raw in keys else default


def ask_float(prompt, default, low, high):
    raw = input(f"  {prompt} [default {default:g}]: ").strip()
    try:
        value = float(raw)
        if not (low <= value <= high):
            raise ValueError
        return value
    except ValueError:
        if raw:
            print(f"  -> Not valid, using {default:g}")
        return default


def ask_int(prompt, default):
    raw = input(f"  {prompt} [default {default}]: ").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else default


def resolve_file_input(raw):
    """Resolve a user-typed file path relative to the project and data folders."""
    raw = str(raw).strip().strip('"').strip("'")
    if not raw:
        return ""
    candidates = [
        raw,
        os.path.expanduser(raw),
        os.path.abspath(raw),
        os.path.join(SCRIPT_DIR, raw),
        os.path.join(DATA_DIR, raw),
        os.path.join(os.getcwd(), raw),
    ]
    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm not in seen and os.path.isfile(norm):
            return norm
        seen.add(norm)
    return raw


def choose_file(title, files):
    """Show the available files and require the user to type the exact filename."""
    print(f"\n  {title}")
    for i, f in enumerate(files, 1):
        print(f"    [{i}] {os.path.relpath(f, SCRIPT_DIR)}")
    while True:
        raw = input("  Type the exact filename (relative path or full path) [default 1]: ")
        raw = raw.strip().strip('"').strip("'")
        if not raw and files:
            return files[0]
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            return files[int(raw) - 1]
        res = resolve_file_input(raw)
        if raw and os.path.isfile(res):
            return res
        print(f"  x '{raw}' is not a valid file name. Type something like 'data/interferents_example.txt'.")


def count_entries(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() and not line.strip().startswith("#"))


def ask_source():
    src = choose("BINDING DATA TYPE", [
        ("1", "Experimental only           - measured Ki only. Tool 1 picks the best-MEASURED OBPs that help:\n"
              "                                    a panel you can trust, if the measured data allow one"),
        ("2", "With predicted (Tanimoto)   - measured + imputed Ki; predictions take part in choosing the OBPs:\n"
              "                                    finds panels mode 1 cannot see, the bets list says what to measure"),
    ])
    args = ["--source", "experimental" if src == "1" else "predicted"]
    if src == "2":
        print("\n  Tanimoto confidence c1 of each imputed cell (0-1). Lower = weaker prediction.")
        print("    0   = keep every prediction")
        print("    0.3 = drop weak predictions (recommended)")
        args += ["--min-confidence", str(ask_float("Minimum confidence", 0.3, 0.0, 1.0))]
    return args


def real_tool1():
    interf_files = sorted(glob.glob(os.path.join(DATA_DIR, "interferents_*.txt")))
    args = ["--data-dir", DATA_DIR] + ask_source()
    print("\n  The interferent file contains names or CAS numbers, one per line.")
    for f in interf_files:
        print(f"    {os.path.basename(f)}: {count_entries(f)} compounds")
    args += ["--interferents", choose_file("Interferent file:", interf_files)]
    print("\n  Noise floor: the smallest residual the sensor can distinguish.")
    args += ["--noise", str(ask_float("Noise floor", 0.05, 1e-9, 1e9))]
    print("\n  Target VOC and starting OBP are asked next.")
    return "scentinel_tool1_specificity.py", args


def real_tool2():
    profile_files = (sorted(glob.glob(os.path.join(DATA_DIR, "vineyard", "*.csv")))
                     + sorted(glob.glob(os.path.join(SCRIPT_DIR, "templates", "*.xlsx")))
                     + sorted(glob.glob(os.path.join(SCRIPT_DIR, "examples", "*vineyard*.csv"))))
    args = ["--data-dir", DATA_DIR] + ask_source()
    print("\n  Profiles: a summary table (CAS | name | mean/SE healthy | mean/SE diseased)")
    print("  or per-sample GC-MS profiles with a 'class' column.")
    args += ["--profiles", choose_file("Profile file:", profile_files)]
    args += ["--n", str(ask_int("Plants per group behind each SE (summary tables)", 3))]
    return "scentinel_tool2_discrimination.py", args


def real_report():
    """PDF report: Tool 1 (both data modes) + Tool 2 (both modes, if a summary table is given)."""
    interf_files = sorted(glob.glob(os.path.join(DATA_DIR, "interferents_*.txt")))
    tables = (sorted(glob.glob(os.path.join(DATA_DIR, "vineyard", "*.csv")))
              + sorted(glob.glob(os.path.join(SCRIPT_DIR, "examples", "tool2_mesa_interferentes", "*.csv"))))
    args = ["--data-dir", DATA_DIR]
    print("\n  The report runs Tool 1 with experimental AND predicted data, and Tool 2 if you give")
    print("  a vineyard summary table (CAS | name | mean/SE healthy | mean/SE diseased).")
    print("\n  The interferent file contains names or CAS numbers, one per line.")
    for f in interf_files:
        print(f"    {os.path.basename(f)}: {count_entries(f)} compounds")
    args += ["--interferents", choose_file("Interferent file:", interf_files)]
    print("\n  Vineyard summary table for Tool 2:")
    for f in tables:
        print(f"    {os.path.relpath(f, SCRIPT_DIR)}")
    while True:
        raw = input("  Type the exact filename (relative path or full path) [Enter = skip Tool 2]: ")
        raw = raw.strip().strip('"').strip("'")
        if not raw:
            args += ["--profiles", ""]
            break
        res = resolve_file_input(raw)
        if os.path.isfile(res):
            args += ["--profiles", res]
            break
        print(f"  x '{raw}' is not a valid file name. Type something like 'data/vineyard/bianca_2017_48hpi.csv'.")
    args += ["--min-confidence", str(ask_float("Predicted mode: minimum Tanimoto confidence", 0.3, 0.0, 1.0))]
    args += ["--noise", str(ask_float("Tool 1 noise floor", 0.05, 1e-9, 1e9))]
    args += ["--max-error", str(ask_float("Tool 2: max share of plants misclassified", 0.05, 1e-4, 0.5))]
    print("\n  Target VOC and starting OBP are asked next. The PDF is saved in the current folder.")
    return "generate_panel_report.py", args


def run_once():
    tool = choose("TOOL", [
        ("1", "Tool 1 - Specificity      - smallest panel so nothing imitates the target VOC"),
        ("2", "Tool 2 - Discrimination   - panel that best separates healthy vs diseased"),
        ("3", "PDF report                - Tool 1 + Tool 2, both data modes, verdicts and lab priorities"),
    ])

    if not os.path.isfile(os.path.join(DATA_DIR, BINDING_FILE_EXPERIMENTAL)):
        print(f"\n  ERROR: '{BINDING_FILE_EXPERIMENTAL}' not found in {DATA_DIR}.")
        print("  Copy the SCENTINEL CSVs there.")
        return
    script, args = {"1": real_tool1, "2": real_tool2, "3": real_report}[tool]()
    print(f"\n{SEP}\n  RUNNING: python {script} {' '.join(args)}\n{SEP}\n")
    module = load_module(script)
    sys.argv = [script] + args
    module.main()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    print("SCENTINEL CODE - OBP panel selection")
    print("Choose the tool; results are printed and saved in ./results/ (PDF report: current folder)")
    while True:
        try:
            run_once()
        except SystemExit as e:          # the tools stop with sys.exit on bad input
            if isinstance(e.code, str):
                print(e.code)
        again = input("\n  Run another analysis? [y/N]: ").strip().lower()
        if again not in ("y", "yes", "s", "si"):
            break


if __name__ == "__main__":
    main()
