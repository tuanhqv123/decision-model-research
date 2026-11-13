import json,glob,sys,collections
d=sys.argv[1]
LIVE=('Camera','Clock','Video','AudioRecorder')
tot=0;c=collections.Counter();resid_tasks=collections.Counter();resid_ep=set();failed_resid=set();allres={}
for f in sorted(glob.glob(f'benchmarks/results/2026-11-12-{d}/androidworld/*/episodes/*.jsonl')):
    if 'abandoned' in f: continue
    rows=[json.loads(l) for l in open(f)]
    name=rows[0]['task_name']; res=rows[-1]
    st=[r for r in rows if r.get('type')=='step']
    seen=collections.defaultdict(set)  # tree -> set of hashes
    for k,(a,b) in enumerate(zip(st,st[1:])):
        key=json.dumps(a.get('elements_compact'))
        seen[key].add(a.get('screen_hash'))
        if not (a.get('screen_hash')!=b.get('screen_hash') and a.get('elements_compact')==b.get('elements_compact')): continue
        tot+=1
        ae=(a.get('actions_executed') or [{}])[0] or {}; act=ae.get('action_type')
        els=a.get('elements_compact') or []
        kb=any(e[0] in ('Switch input method','Voice input') or (e[1]=='FrameLayout' and e[0] in ('Q','W','E')) for e in els)
        recur = b.get('screen_hash') in seen[key]
        if recur: cat='next hash already seen with this same tree (blink/alternation)'
        elif act=='wait': cat='prev action wait'
        elif any(t in name for t in LIVE): cat='live-content task'
        elif kb: cat='keyboard on screen (IME/caret)'
        elif act=='scroll': cat='scroll (possible list end)'
        else:
            cat='residual (consistent with lag)'; resid_tasks[name]+=1; resid_ep.add(name)
            if not res.get('success'): failed_resid.add(name)
        c[cat]+=1
print(d,tot)
for k,v in c.most_common(): print('  %-62s %d'%(k,v))
print('  residual episodes',len(resid_ep),'failed',len(failed_resid), resid_tasks.most_common(12))
