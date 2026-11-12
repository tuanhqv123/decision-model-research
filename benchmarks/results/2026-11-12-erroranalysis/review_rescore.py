import json,sys,math
sys.path.insert(0,'~/Workspace/research/benchmarks/train')
from score_actest import to_infigui_pred
from vendor.infigui_eval import evaluate_android_control_action
R='~/Workspace/research/benchmarks/results/'
def load_ps(p): return {(r['episode_id'],r['step']):r for r in map(json.loads,open(p))}
v3=load_ps(R+'2026-11-11-v3/actest/perstep-v3_s0_ckpt_1824_actest_v3.jsonl')
v4=load_ps(R+'2026-11-12-v4/actest/perstep-v4_s0_ckpt_2482_actest_v4.jsonl')
# gt identical across v3/v4 perstep?
print('gt identical', sum(v3[k]['gt']==v4[k]['gt'] for k in v3), len(v3))
def rescore(evalp, W=1080, H=2400):
    out={}
    for r in map(json.loads,open(evalp)):
        k=(int(r['episode_id']),int(r['step']))
        p=to_infigui_pred(r['pred'],r.get('value'),r.get('pt'))
        t,e=evaluate_android_control_action(p, v3[k]['gt'], W,H,W,H, pred_type='abs_resized', gt_type='abs_resized')
        out[k]=bool(e)
    return out
r4=rescore(R+'2026-11-12-v4/actest/eval-v4_s0_ckpt_2482_actest_v4-VA-val.jsonl')
print('v4 rescored', sum(r4.values()), 'agree with perstep', sum(r4[k]==v4[k]['exact_match'] for k in r4))
m3=rescore(R+'2026-11-12-v4/actest/merge_a03_actest.jsonl')
m5=rescore(R+'2026-11-12-v4/actest/merge_a05_actest.jsonl')
print('merge03', sum(m3.values()), 'merge05', sum(m5.values()))
for name,m in (('m03',m3),('m05',m5)):
    w=sum(m[k] and not v3[k]['exact_match'] for k in m); l=sum(v3[k]['exact_match'] and not m[k] for k in m)
    z=(w-l)/math.sqrt(w+l); print(name,'vs v3: wins',w,'losses',l,'z',round(z,2),'diff',w-l)
# vs v4
w=sum(m3[k] and not v4[k]['exact_match'] for k in m3); l=sum(v4[k]['exact_match'] and not m3[k] for k in m3); print('m03 vs v4', w,l)
json.dump({str(k):v for k,v in m3.items()}, open('/private/tmp/claude-501/-Users-[user]-Workspace-research/5b003b77-1d13-48ff-be88-93d75a03034c/scratchpad/m3.json','w'))
