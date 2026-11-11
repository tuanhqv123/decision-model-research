#!/usr/bin/env python3
"""POST-HOC: failure taxonomy of the online episodes (task-completion tasks). Run from this directory: python3 bench_failures.py r1s0 r1s1 b0prime.
The task -> app map and the "needs typing" rule are HEURISTICS (regexes on the goal); the open_app check is exact (see appresolve.json)."""
import json,collections,re,sys
ans=set(open("../../aw/hostscripts/aw116_answer_only.txt").read().split())
APP={"Markor":"markor","SimpleSms":"sms","Expense":"expense","Recipe":"broccoli","SimpleCalendar":"calendar","Contacts":"contacts",
 "Clock":"clock","AudioRecorder":"audio recorder","Camera":"camera","Files":"files","Osm":"osmand","Retro":"retro","Vlc":"vlc",
 "Browser":"chrome","System":"settings","Turn":"settings","SimpleDraw":"draw","Tasks":"tasks","SportsTracker":"opentracks","Notes":"joplin",
 "Expense":"expense","Save":"","Open":""}
def expected(t):
    for k,v in APP.items():
        if t.startswith(k): return v
    return None
for arm in sys.argv[1:]:
    ex=json.load(open(arm+"_export.json")); ex=[e for e in ex if not e.get("setup_error") and not e.get("unscored")]
    tc=[e for e in ex if e["task"] not in ans]
    fail=[e for e in tc if not (e["score"] or 0)>0.5]
    print("==",arm,"task-completion episodes",len(tc),"failed",len(fail))
    # open_app analysis
    oa=[(e["task"],s.get("app") or "") for e in ex for s in e["steps"] if s["action"]=="open_app"]
    bad=[a for t,a in oa if a=="" or len(a.split())>3]
    print("  open_app: %d, empty name %d, long span (>3 words) %d: %s"%(len(oa),sum(a=="" for t,a in oa),sum(len(a.split())>3 for t,a in oa),collections.Counter(bad).most_common(4)))
    wrong=0;n=0
    for e in tc:
        exp=expected(e["task"]); first=[s for s in e["steps"] if s["action"]=="open_app"]
        if not exp or not first: continue
        n+=1; wrong+= exp not in (first[0].get("app") or "").lower().replace(" ","")  and exp.replace(" ","") not in (first[0].get("app") or "").lower().replace(" ","")
    print("  first open_app names the task's app (heuristic map): wrong in %d of %d task-completion episodes"%(wrong,n))
    clicks=[s for e in ex for s in e["steps"] if s["action"]=="click" and s.get("ui_changed") is not None]
    print("  clicks with no visible effect: %d/%d = %.0f%%"%(sum(not s["ui_changed"] for s in clicks),len(clicks),100*sum(not s["ui_changed"] for s in clicks)/len(clicks)))
    acts=collections.Counter(s["action"] for e in ex for s in e["steps"]); tot=sum(acts.values())
    print("  action mix:",{k:"%.1f%%"%(100*v/tot) for k,v in acts.most_common()})
    # typing need: goal contains text after ':' or quotes or 'named'/'titled'/'with ... text'
    need=[e for e in tc if re.search(r"(named|titled|text|message|with the|description|name of|called|:)",e["goal"],re.I) and re.search(r"(Create|Add|Send|Reply|Write|Rename|Edit|Record|Save|new)",e["goal"])]
    typed=[e for e in need if any(s["action"]=="type_text" for s in e["steps"])]
    print("  tasks that (heuristically) need typing %d; model typed at least once in %d; success %d"%(len(need),len(typed),sum((e["score"] or 0)>0.5 for e in need)))
    # failure taxonomy (first-match order)
    cat=collections.Counter(); ex_cat=collections.defaultdict(list)
    for e in fail:
        st=e["steps"]; acts=[s["action"] for s in st]
        first=[s for s in st if s["action"]=="open_app"]
        exp=expected(e["task"])
        nm=(first[0].get("app") or "").lower().replace(" ","") if first else ""
        if exp and first and exp.replace(" ","") not in nm: c="wrong or unnamed app"
        elif e in need and "type_text" not in acts: c="never typed (task needs text)"
        elif e["declared"] and len(st)<=0.5*e["budget"]: c="said done early (<=half budget)"
        elif e["declared"]: c="said done late, wrong"
        else:
            # loop: >=3 consecutive same action with no effect
            loop=any(st[i]["action"]==st[i+1]["action"]==st[i+2]["action"] and st[i].get("ui_changed") is False and st[i+1].get("ui_changed") is False for i in range(len(st)-2))
            c="ran out of steps, stuck in a no-effect loop" if loop else "ran out of steps, moving"
        cat[c]+=1; ex_cat[c].append(e["task"])
    for c,v in cat.most_common(): print("   %-45s %2d  e.g. %s"%(c,v,ex_cat[c][:4]))
