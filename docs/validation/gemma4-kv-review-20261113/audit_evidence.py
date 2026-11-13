#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import collections
import copy
import gzip
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Audit the recorded full-scope KV16 evidence without a device.")
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.repo.resolve()
args.output.parent.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('evidence', root / 'tools/kv16/compare_device_evidence.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
directory = root / 'docs/validation/kv16-npu-device-20261111/encode-opt'
left = json.loads((directory / 'baseline-evidence-clz3.json').read_text())
right = json.loads((directory / 'codec-evidence-clz3.json').read_text())
report = {'source_revision': 'a185c0b69b242e7510c1d807a473510e585090b2',
          'recorded_identity': right['identity'],
          'original_status': mod.compare(left, right)['status'],
          'events': [{k: v for k, v in e.items() if k in ('id', 'name', 'position')} for e in right['events']]}
mutated = copy.deepcopy(right)
mutated['runs'][-1]['ids'] = [999999]
mutated['runs'][-1]['position'] += 123
report['mutated_final_run_status'] = mod.compare(left, mutated)['status']
mutated = copy.deepcopy(right)
mutated['identity']['code_revision'] = 'unknown'
other = copy.deepcopy(left)
other['identity']['code_revision'] = 'unknown'
report['both_revision_unknown_status'] = mod.compare(other, mutated)['status']
for mode in ('baseline', 'codec'):
    rows = [json.loads(x) for x in gzip.open(directory / (mode + '-trace-clz3.jsonl.gz'), 'rt') if x.strip()]
    report[mode + '_trace'] = {
        'events': dict(collections.Counter(r['event'] for r in rows)),
        'calls': [{k: r[k] for k in ('event', 'from', 'position', 'plane', 'chunk')} for r in rows],
    }
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
