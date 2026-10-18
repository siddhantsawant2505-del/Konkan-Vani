"""
Structural validation of the LaTeX deliverables.
================================================
Checks every .tex file we ship for the mistakes that break a compile:

  * balanced braces
  * matched \\begin/\\end environments
  * every \\input target exists
  * every referenced figure exists on the report's graphics path
    (the hand-added screenshots are recognised as intentional placeholders)
  * every \\ref has a \\label (main report + included tables)
  * no non-ASCII text — the reports are compiled with pdfLaTeX + inputenc,
    which cannot typeset Devanagari or typographic dashes

Usage:  python -X utf8 src/verify_report.py
"""

import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

REPORT = "konkan_vani_report.tex"
CHAPTER = "results/chapter5_results.tex"
# Screenshots the authors drop in by hand; the \safefig macro draws a
# placeholder until they exist, so their absence is expected.
EXPECTED_MISSING = {"logo.png", "Search Page.png", "Result Page.png"}

failures = []


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def env_counts(tex):
    begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", tex))
    ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", tex))
    return {k: (begins[k], ends[k]) for k in set(begins) | set(ends) if begins[k] != ends[k]}


def graphics_dirs(tex):
    match = re.search(r"\\graphicspath\{(.*?)\}\s*$", tex, re.M)
    dirs = re.findall(r"\{([^}]*)\}", match.group(1)) if match else []
    return [d for d in dirs if d] or ["./"]


def check_tex(path, label, search_dirs, extra_labels=()):
    tex = open(path, encoding="utf-8").read()
    print(f"\n── {label} ({path}) ──")
    check(f"{label}: braces balanced", tex.count("{") == tex.count("}"),
          f"{tex.count('{')} open vs {tex.count('}')} close")
    mismatch = env_counts(tex)
    check(f"{label}: environments matched", not mismatch, str(mismatch))

    for inc in re.findall(r"\\input\{([^}]+)\}", tex):
        target = inc if inc.endswith(".tex") else inc + ".tex"
        check(f"{label}: input exists: {target}", os.path.exists(target))

    figs = set(re.findall(r"\\safefig\{([^}]+)\}", tex)) | set(
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    for fig in sorted(figs):
        if "#" in fig:            # parameter of the \safefig definition
            continue
        found = any(os.path.exists(os.path.join(d, fig)) for d in search_dirs)
        if fig in EXPECTED_MISSING:
            check(f"{label}: screenshot still a placeholder: {fig}", not found,
                  "file exists — the placeholder simply will not be used")
            continue
        check(f"{label}: figure present: {fig}", found, f"searched {search_dirs}")

    if re.search(r"\\ref\{", tex):
        labels = set(re.findall(r"\\label\{([^}]+)\}", tex)) | set(extra_labels)
        for ref in sorted(set(re.findall(r"\\ref\{([^}]+)\}", tex))):
            check(f"{label}: ref resolved: {ref}", ref in labels)

    leftover = sorted({ch for ch in tex if ord(ch) > 127})
    check(f"{label}: ASCII only (pdfLaTeX + inputenc)", not leftover, str(leftover))
    return figs


report = open(REPORT, encoding="utf-8").read()
dirs = graphics_dirs(report)
input_labels = set()
for inc in re.findall(r"\\input\{([^}]+)\}", report):
    target = inc if inc.endswith(".tex") else inc + ".tex"
    if os.path.exists(target):
        input_labels |= set(re.findall(r"\\label\{([^}]+)\}",
                                       open(target, encoding="utf-8").read()))

figs_main = check_tex(REPORT, "main report", dirs, input_labels)
figs_chapter = check_tex(CHAPTER, "chapter 5", ["./"], input_labels)

# Included table files are typeset by the same engine, so check them too.
for path in ("results/tables.tex", "results/tables_composition.tex"):
    if not os.path.exists(path):
        continue
    tex = open(path, encoding="utf-8").read()
    print(f"\n-- included table file: {path} --")
    check(f"{path}: braces balanced", tex.count("{") == tex.count("}"))
    mismatch = env_counts(tex)
    check(f"{path}: environments matched", not mismatch, str(mismatch))
    leftover = sorted({ch for ch in tex if ord(ch) > 127})
    check(f"{path}: ASCII only", not leftover, str(leftover))

print()
if failures:
    print(f"RESULT: {len(failures)} issue(s)")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"RESULT: all LaTeX files structurally OK "
      f"({len(figs_main)} figures in the main report, {len(figs_chapter)} in chapter 5)")
