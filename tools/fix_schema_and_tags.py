#!/usr/bin/env python3
"""
Fix schema_version to 3.4 and ensure category tags present in marker YAMLs under markers.
Non-destructive: creates .bak copy before modifying.
Writes a report to tools/fix_schema_and_tags_report.json
Scans recursively only the subfolders: ATO, SEM, CLU, CLU_INTUITION and skips NEG_EXAMPLES.yaml.
"""
import os,re,json,shutil

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
markers_root = os.path.join(workspace_root, 'markers')
expected_schema = '3.4'

mapping = {
    'ATO_': 'atomic',
    'SEM_': 'semantic',
    'CLU_INTUITION': 'clu_intuition',
    'CLU_': 'cluster',
    'MEMA_': 'meta',
    'META_': 'meta'
}

# only process these subfolders under markers
allowed_subdirs = set(['ATO', 'SEM', 'CLU', 'CLU_INTUITION', 'MEMA', 'META'])

report = {'modified': [], 'skipped': [], 'errors': [], 'checked_files': 0}

for root, dirs, files in os.walk(markers_root):
    # determine if this directory is under an allowed subdir
    rel = os.path.relpath(root, markers_root)
    parts = rel.split(os.sep)
    if parts[0] == '.' or parts[0] == '..':
        parts = []
    if parts and parts[0] not in allowed_subdirs:
        # skip directories not in the allowed set
        continue
    for fn in files:
        if not fn.lower().endswith(('.yaml', '.yml')):
            continue
        if fn == 'NEG_EXAMPLES.yaml' or fn == 'NEG_EXAMPLES.yml':
            report['skipped'].append(os.path.join(root, fn))
            continue
        fp = os.path.join(root, fn)
        report['checked_files'] += 1
        try:
            txt = open(fp, 'r', encoding='utf-8').read()
        except Exception as e:
            report['errors'].append({'file': fp, 'error': f'read failed: {e}'})
            continue
        original = txt
        changed = False
        # find id
        m_id = re.search(r'^\s*id:\s*(\S+)', txt, flags=re.MULTILINE)
        file_id = m_id.group(1).strip() if m_id else None
        # fix schema_version
        m_schema = re.search(r'^\s*schema_version:\s*"?([0-9\.]+)"?', txt, flags=re.MULTILINE)
        if m_schema:
            sv = m_schema.group(1).strip()
            if sv != expected_schema:
                txt = re.sub(r'^\s*schema_version:.*$', f'schema_version: "{expected_schema}"', txt, count=1, flags=re.MULTILINE)
                changed = True
        else:
            # insert schema_version before EOF
            txt = txt.rstrip() + '\n\nschema_version: "' + expected_schema + '"\n'
            changed = True
        # determine desired tag
        desired = None
        if file_id:
            for p, tag in mapping.items():
                if file_id.startswith(p):
                    desired = tag
                    break
        # ensure tags include desired
        if desired:
            m_tags = re.search(r'^\s*tags\s*:\s*\[(.*?)\]', txt, flags=re.MULTILINE|re.DOTALL)
            if m_tags:
                inner = m_tags.group(1)
                tokens = [t.strip().strip('"\'') for t in inner.split(',') if t.strip()]
                if desired not in tokens:
                    tokens.append(desired)
                    new_inner = ', '.join([('"'+t+'"') for t in tokens])
                    txt = re.sub(r'^\s*tags\s*:\s*\[.*?\]', f'tags: [{new_inner}]', txt, flags=re.MULTILINE|re.DOTALL)
                    changed = True
            else:
                # try block tags
                m_block = re.search(r'^\s*tags\s*:\s*\n((?:\s*-.*\n)+)', txt, flags=re.MULTILINE)
                if m_block:
                    block = m_block.group(1)
                    existing = [re.sub(r'^\s*-\s*', '', l).strip() for l in block.splitlines() if l.strip()]
                    if desired not in existing:
                        block = block.rstrip() + f'\n  - {desired}\n'
                        txt = re.sub(r'(^\s*tags\s*:\s*\n(?:\s*-.*\n)+)', 'tags:\n' + block, txt, flags=re.MULTILINE)
                        changed = True
                else:
                    # insert tags before schema_version
                    txt = re.sub(r'(\nschema_version:)', f'\ntags: ["{desired}"]\n\1', txt, count=1, flags=re.MULTILINE)
                    changed = True
        # write backup and update file if changed
        if changed and txt != original:
            try:
                shutil.copy2(fp, fp + '.bak')
                with open(fp, 'w', encoding='utf-8') as fh:
                    fh.write(txt)
                report['modified'].append(fp)
            except Exception as e:
                report['errors'].append({'file': fp, 'error': f'write failed: {e}'})
        else:
            report['skipped'].append(fp)

# write report
outp = os.path.join(os.path.dirname(__file__), 'fix_schema_and_tags_report.json')
with open(outp, 'w', encoding='utf-8') as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

print(json.dumps({'modified': len(report['modified']), 'skipped': len(report['skipped']), 'errors': len(report['errors']), 'checked_files': report['checked_files']}))
