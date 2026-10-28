#!/usr/bin/env python3
"""Export held-out, real-dataset feature vectors/labels for csp_lda_core RTL testbench."""
import argparse, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('model', help='results_fb/model_s01.npz')
p.add_argument('--out', default='dataset_vectors')
p.add_argument('--max-trials', type=int, default=100, help='0 means all trials')
p.add_argument('--feature-frac', type=int, default=8)
p.add_argument('--weight-frac', default='auto', help="fraction bits for weights/bias, or 'auto' (largest value <=14 that avoids clipping)")
a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
d=np.load(a.model, allow_pickle=True)
if 'test_features' not in d.files:
    raise SystemExit('Model has no test_features. Use the updated bci_csp_lda_fb.py from this package, then retrain.')
F=np.asarray(d['test_features'], dtype=np.float64); y=np.asarray(d['test_labels'], dtype=np.int64).reshape(-1)
W=np.asarray(d['lda_W'], dtype=np.float64); b=np.asarray(d['lda_b'], dtype=np.float64).reshape(-1)
if F.ndim != 2 or F.shape[1] != 96 or W.shape != (96,4) or b.shape != (4,):
    raise SystemExit(f'Unexpected shapes: features={F.shape}, W={W.shape}, b={b.shape}; expected (N,96),(96,4),(4,)')
if len(y) != len(F): raise SystemExit('Feature/label count mismatch')
if str(a.weight_frac) == 'auto':
    need = max(float(np.abs(W).max()), 1e-9)
    a.weight_frac = max(0, min(14, 15 - int(np.ceil(np.log2(need + 1e-12))) - 1))
    while a.weight_frac > 0 and np.abs(W).max() * (1 << a.weight_frac) > 32767: a.weight_frac -= 1
else:
    a.weight_frac = int(a.weight_frac)
if a.max_trials > 0: F, y = F[:a.max_trials], y[:a.max_trials]
def signed_hex(v,bits): return f'{int(v)&((1<<bits)-1):0{bits//4}X}'
def quant(arr, frac, bits):
    raw=np.rint(np.asarray(arr)*(1<<frac)); lo=-(1<<(bits-1)); hi=(1<<(bits-1))-1
    q=np.clip(raw,lo,hi).astype(np.int64); return q, int(np.count_nonzero(raw != q))
Fq,nf=quant(F,a.feature_frac,16); Wq,nw=quant(W,a.weight_frac,16); bq,nb=quant(b,a.weight_frac,48)
# Match RTL arithmetic: each signed feature*weight product is arithmetic-shifted before summing.
scores=np.tile(bq.reshape(1,4),(len(Fq),1)).astype(np.int64)
for i in range(96):
    for c in range(4): scores[:,c] += (Fq[:,i] * Wq[i,c]) >> a.feature_frac
ref=scores.argmax(axis=1).astype(np.int64)
with (out/'test_features.mem').open('w') as f:
    for row in Fq:
        for v in row: f.write(signed_hex(v,16)+'\n')
with (out/'test_labels.mem').open('w') as f:
    for v in y: f.write(f'{int(v)&3:X}\n')
with (out/'expected_predictions.mem').open('w') as f:
    for v in ref: f.write(f'{int(v)&3:X}\n')
with (out/'lda_weights.mem').open('w') as f:
    for i in range(96):
        for c in range(4): f.write(signed_hex(Wq[i,c],16)+'\n')
with (out/'lda_bias.mem').open('w') as f:
    for v in bq: f.write(signed_hex(v,48)+'\n')
info={'source_model':str(a.model),'test_source':str(d['test_source'].item()) if 'test_source' in d else 'unknown',
      'trials':int(len(y)),'features_per_trial':96,'classes':['left hand','right hand','feet','tongue'],
      'feature_fraction_bits':a.feature_frac,'weight_fraction_bits':a.weight_frac,
      'feature_values_clipped':nf,'weight_values_clipped':nw,'bias_values_clipped':nb,
      'quantized_software_accuracy_vs_dataset_labels':float(np.mean(ref==y)),
      'note':'Features and labels come from held-out dataset trials processed by the notebook pipeline.'}
(out/'dataset_testbench_info.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
if nf or nw or nb: print('WARNING: values were clipped during quantization; RTL accuracy may differ from the model. Lower --feature-frac or --weight-frac.')
print('Created dataset feature/label files and LDA ROMs in',out)
