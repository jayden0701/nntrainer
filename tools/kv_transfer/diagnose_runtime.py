"""Small CPU cross-runtime KV/teacher-forcing diagnostic; no mapper fitting."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import NativeScorer, model_spec
from snapshot import Snapshot, create_snapshot
from mapper import Mapping, apply
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parents[2]


def stats(a, b):
    a, b = a.astype('float64'), b.astype('float64')
    delta = a - b
    return {'max_abs': float(abs(delta).max()), 'rmse': float(np.sqrt(np.mean(delta**2))),
            'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(b), 1e-30)),
            'cosine': float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-hf', type=Path, required=True)
    parser.add_argument('--target-hf', type=Path, required=True)
    parser.add_argument('--case-set', choices=('short', '128'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'artifacts/ridge-paper-data/bench-manifest-128.json').read_text())
    examples = ([next(e for e in manifest['items'] if e['task']=='wikitext2')] if args.case_set == '128' else [next(e for e in manifest['items'] if e['task']==task) for task in ('arc_challenge','hellaswag')])
    configs = [('0.6b', args.source_hf, 'Qwen/Qwen3-0.6B'),
               ('1.7b',args.target_hf,'Qwen/Qwen3-1.7B')]
    report={'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__},
            'reference':'CPU FP32 eager attention, no tokenizer/chat template, identical IDs', 'models':{}}
    for short, hf_dir, model_id in configs:
        native_dir=ROOT/f'artifacts/models/qwen3-{short}-fp32'
        assert (hf_dir/'config.json').read_bytes()==(native_dir/'config.json').read_bytes()
        model=AutoModelForCausalLM.from_pretrained(str(hf_dir),dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval()
        scorer=NativeScorer(ROOT/'build-cpu/Applications/CausalLM/nntr_causallm',native_dir,model_spec(native_dir,model_id),4,1)
        work=out/short;work.mkdir(exist_ok=True); jobs=[];records=[]
        for e in examples:
            case_id=e['id']; scratch=work/case_id;scratch.mkdir(exist_ok=True)
            ids=e['context_ids'][:128]; targets=e['candidates'][e.get('gold',0)]['ids'][:8]
            all_ids=ids+targets[:-1]
            with torch.inference_mode():
                hf=model(torch.tensor([all_ids]),use_cache=True)
                rows=hf.logits[0,len(ids)-1:len(ids)-1+len(targets)].double().log_softmax(-1)
                ll=rows[torch.arange(len(targets)),torch.tensor(targets)].tolist()
                top_id=int(rows[0].argmax()); top_lp=float(rows[0,top_id])
            snapshot=scratch/'native-kv'
            if not snapshot.exists():scorer.export(scratch,scratch/'export.log',ids,snapshot)
            native=Snapshot(snapshot); layers=[]
            assert native.token_ids == tuple(ids)
            for l, cache in enumerate(hf.past_key_values.layers):
                k=cache.keys[0,:,:len(ids)].transpose(0,1).float().numpy()
                v=cache.values[0,:,:len(ids)].transpose(0,1).float().numpy()
                layers.append({'layer':l,'K':stats(np.array(native.plane('keys')[l]),k),'V':stats(np.array(native.plane('values')[l]),v),
                               'K_vs_rounded_reference':stats(np.array(native.plane('keys')[l]),k.astype('float16').astype('float32')),
                               'V_vs_rounded_reference':stats(np.array(native.plane('values')[l]),v.astype('float16').astype('float32'))})
            record={'id':case_id,'context_ids':ids,'targets':targets,'hf_logprobs':ll,'hf_top1':top_id,'hf_top1_logprob':top_lp,'layers':layers}
            records.append(record)
            job={'id':case_id,'context_ids':ids,'candidates':[{'ids':targets},{'ids':[top_id]}]}
            jobs.append((job,scratch,{'cold':None}))
            del hf
        mapped_ref = None
        if short == '1.7b' and args.case_set == '128':
            source = Snapshot(out/'0.6b'/case_id/'native-kv')
            trimmed = work/'source-prefix'
            if not trimmed.exists():
                with create_snapshot(trimmed, source.spec, ids[:-1], range(len(ids)-1)) as planes:
                    for channel in planes: planes[channel][:] = source.plane(channel)[:, :-1]
            mapped_path=work/'mapped-cache'
            if not mapped_path.exists():
                apply(Mapping(ROOT/'artifacts/public-ridge-k4-n200'), Snapshot(trimmed), mapped_path, block_size=128, memory_mib=1024)
            mapped=Snapshot(mapped_path)
            cache=DynamicCache()
            for layer in range(28):
                planes=[]
                for channel in ('keys','values'):
                    a=np.array(mapped.plane(channel)[layer]).astype('float16').astype('float32')
                    planes.append(torch.from_numpy(a.transpose(1,0,2)[None].copy()))
                cache.update(*planes, layer)
            with torch.inference_mode():
                mapped_out=model(torch.tensor([ids[-1:]+targets[:-1]]), past_key_values=cache, use_cache=True)
                lp=mapped_out.logits[0].double().log_softmax(-1)
                mapped_ref=lp[torch.arange(len(targets)),torch.tensor(targets)].tolist()
            del cache,mapped_out
        del model;gc.collect()
        scores,seconds=scorer.score(work,work/'native-score.log',jobs,persistent=True)
        for record in records:
            native=scores[record['id']]['cold'];lp=native[0]['token_logprobs'];record['native_logprobs']=lp
            record['logprob_max_abs_error']=float(np.max(np.abs(np.array(lp)-record['hf_logprobs'])))
            record['native_hf_top1_logprob']=native[1]['token_logprobs'][0]
        if mapped_ref is not None:
            mapped_work=work/'mapped-score';mapped_work.mkdir(exist_ok=True)
            mapped_case={'id':case_id,'context_ids':ids,'candidates':[{'ids':targets}]}
            native_map,_=scorer.score(mapped_work,mapped_work/'score.log',[(mapped_case,mapped_work,{'transfer':mapped})],persistent=True)
            mapped_native=native_map[case_id]['transfer'][0]['token_logprobs']
            report['mapped_consumer_check']={'hf_logprobs':mapped_ref,'native_logprobs':mapped_native,
                'prefix_tokens':mapped.tokens,'max_abs_logprob_error':float(np.max(np.abs(np.array(mapped_ref)-mapped_native))),
                'hf_mapped_minus_cold':(np.array(mapped_ref)-records[0]['hf_logprobs']).tolist()}
        report['models'][short]=records
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(short,[(x['id'],x['logprob_max_abs_error'],x['layers'][0]['K']['relative_l2'],x['layers'][-1]['K']['relative_l2']) for x in records],flush=True)


if __name__=='__main__':main()
