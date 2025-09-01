#!/usr/bin/env python3
"""
Scan SEM markers in TransRa._Marker for composed_of ATO_ dependencies.
If an ATO_ marker is missing in the target, try to copy it from one of the LD3.4 source directories.
Writes a JSON report to tools/fill_missing_report.json and prints a short summary.
"""
import os, sys, re, json, shutil

target_dir = "/Users/benjaminpoersch/Projekte/XEXPERIMENTE/TransRapport_V2/TransRa._Marker"
sources = [
    "/Users/benjaminpoersch/Projekte/XEXPERIMENTE/LD3.4_Marker_5.0/Marker_5.0",
    "/Users/benjaminpoersch/Projekte/XEXPERIMENTE/LD3.4_Marker_5.0",
    "/Users/benjaminpoersch/Projekte/XEXPERIMENTE/OUTput_Marker"
]

report = {"scanned_sem_files": 0, "required_atomics": [], "already_present": [], "copied": [], "not_found": []}

def find_ato_ids_in_text(text):
    # Conservative: extract ATO_ tokens that are likely in composed_of lists
    # We look for ATO_ tokens anywhere in the file (practical and robust)
    return set(re.findall(r"\b(ATO_[A-Z0-9_]+)\b", text))

# list SEM files in target_dir
sem_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.startswith('SEM_') and f.endswith('.yaml')]
for sem in sem_files:
    try:
        with open(sem, 'r', encoding='utf-8') as fh:
            txt = fh.read()
    except Exception as e:
        print(f"WARN: could not read {sem}: {e}", file=sys.stderr)
        continue
    report['scanned_sem_files'] += 1
    atos = find_ato_ids_in_text(txt)
    for ato in sorted(atos):
        if ato not in report['required_atomics']:
            report['required_atomics'].append(ato)

# normalize
report['required_atomics'].sort()

for ato in report['required_atomics']:
    target_path = os.path.join(target_dir, f"{ato}.yaml")
    if os.path.exists(target_path):
        report['already_present'].append(ato)
        continue
    # try to find in sources
    found = None
    # try exact filename in sources
    for s in sources:
        candidate = os.path.join(s, f"{ato}.yaml")
        if os.path.exists(candidate):
            found = candidate
            break
    # recursive search if not found
    if not found:
        for s in sources:
            for root, dirs, files in os.walk(s):
                for fn in files:
                    if fn.lower().endswith('.yaml'):
                        # quick filename match
                        if ato.lower() in fn.lower():
                            candidate = os.path.join(root, fn)
                            found = candidate
                            break
                        # otherwise inspect content for id: ATO_
                        fp = os.path.join(root, fn)
                        try:
                            with open(fp, 'r', encoding='utf-8') as fh:
                                content = fh.read()
                        except Exception:
                            continue
                        if re.search(r"^id:\s*" + re.escape(ato) + r"\b", content, flags=re.MULTILINE):
                            found = fp
                            break
                if found:
                    break
            if found:
                break
    if found:
        # copy to target
        try:
            shutil.copy2(found, target_path)
            report['copied'].append({"ato": ato, "from": found, "to": target_path})
        except Exception as e:
            report['not_found'].append({"ato": ato, "error": f"failed to copy: {e}"})
    else:
        report['not_found'].append(ato)

# write report
outp = os.path.join(os.path.dirname(__file__), 'fill_missing_report.json')
try:
    with open(outp, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
except Exception as e:
    print(f"ERROR: could not write report {outp}: {e}", file=sys.stderr)

# print concise summary
print(json.dumps({
    'scanned_sem_files': report['scanned_sem_files'],
    'required_atomics_count': len(report['required_atomics']),
    'already_present': len(report['already_present']),
    'copied': len(report['copied']),
    'not_found': len(report['not_found'])
}))

# also print lists in a compact form
if report['copied']:
    print('\nCOPIED:')
    for c in report['copied']:
        print(f"{c['ato']} <- {c['from']}")
if report['not_found']:
    print('\nMISSING:')
    for m in report['not_found']:
        if isinstance(m, dict):
            print(f"{m.get('ato','?')} error: {m.get('error')}")
        else:
            print(m)

sys.exit(0)
