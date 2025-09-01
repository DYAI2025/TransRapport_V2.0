#!/usr/bin/env python3
"""
Validate marker YAML files for LEAN.DEEP_3.4 conformity (best-effort without PyYAML).
Checks per-file:
 - presence of id
 - presence of schema_version (expects '3.4' ideally)
 - presence of at least one of pattern/frame/examples/composed_of
 - tags include a category matching id prefix (ATO_/SEM_/CLU_/MEMA_/CLU_INTUITION)
 - for SEM/CLU files, check composed_of ATO_ references exist in target marker dirs
Writes JSON report to tools/lean_deep_validation.json
"""
import os,re,json

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
marker_dirs = [
    os.path.join(workspace_root, 'TransRa._Marker'),
    os.path.join(workspace_root, 'markers')
]
expected_schema = '3.4'

report = {'checked_files':0, 'issues':[], 'missing_atomic_refs':{}, 'summary':{}}
all_files = []
for d in marker_dirs:
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        if fn.lower().endswith('.yaml'):
            all_files.append(os.path.join(d, fn))

# collect set of ids present
ids_present = set()
for fp in all_files:
    try:
        txt = open(fp, 'r', encoding='utf-8').read()
    except Exception:
        continue
    m = re.search(r"^id:\s*(\S+)", txt, flags=re.MULTILINE)
    if m:
        ids_present.add(m.group(1).strip())

for fp in all_files:
    report['checked_files'] += 1
    try:
        txt = open(fp, 'r', encoding='utf-8').read()
    except Exception as e:
        report['issues'].append({'file':fp,'error':f'cannot read: {e}'})
        continue
    issues = []
    m_id = re.search(r"^id:\s*(\S+)", txt, flags=re.MULTILINE)
    if not m_id:
        issues.append('missing id')
        file_id = None
    else:
        file_id = m_id.group(1).strip()
    m_schema = re.search(r"^schema_version:\s*\"?([0-9\.]+)\"?", txt, flags=re.MULTILINE)
    if not m_schema:
        issues.append('missing schema_version')
    else:
        sv = m_schema.group(1).strip()
        if sv != expected_schema:
            issues.append(f'schema_version mismatch (found {sv}, expected {expected_schema})')
    # check for content fields
    if not re.search(r"^pattern:\s*$|^frame:\s*$|^examples?:|^composed_of:\s*$", txt, flags=re.MULTILINE):
        issues.append('no pattern/frame/examples/composed_of found')
    # tags/category
    prefix = None
    if file_id:
        if file_id.startswith('ATO_'):
            prefix = 'ATO'
        elif file_id.startswith('SEM_'):
            prefix = 'SEM'
        elif file_id.startswith('CLU_'):
            prefix = 'CLU'
        elif file_id.startswith('MEMA_') or file_id.startswith('META_'):
            prefix = 'META'
        elif file_id and file_id.startswith('CLU_INTUITION'):
            prefix = 'CLU_INTUITION'
    if prefix:
        # check tags include matching category token
        m_tags = re.search(r"^tags:\s*\[(.*?)\]", txt, flags=re.MULTILINE|re.DOTALL)
        has_cat = False
        if m_tags:
            tags = m_tags.group(1)
            if re.search(prefix.lower(), tags, flags=re.IGNORECASE) or re.search(prefix, tags):
                has_cat = True
        if not has_cat:
            issues.append(f'tags missing {prefix.lower()}/category')
    # composed_of references
    if re.search(r"^composed_of:\s*$", txt, flags=re.MULTILINE):
        # collect ATO_ tokens
        refs = set(re.findall(r"\b(ATO_[A-Z0-9_]+)\b", txt))
        missing = [r for r in refs if r not in ids_present]
        if missing:
            report['missing_atomic_refs'][fp] = missing
            issues.append(f'composed_of refs missing: {len(missing)}')
    if issues:
        report['issues'].append({'file': fp, 'id': file_id, 'issues': issues})

# build summary
report['summary']['total_files'] = len(all_files)
report['summary']['files_with_issues'] = len(report['issues'])
report['summary']['files_missing_atomic_refs'] = len(report['missing_atomic_refs'])

outp = os.path.join(os.path.dirname(__file__), 'lean_deep_validation.json')
with open(outp, 'w', encoding='utf-8') as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

print(json.dumps({'total_files': report['summary']['total_files'], 'issues': report['summary']['files_with_issues'], 'missing_atomic_ref_files': report['summary']['files_missing_atomic_refs']}))
