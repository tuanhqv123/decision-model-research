"""Is the right answer in the model's 2nd/3rd choice? (AC-High, ckpt_1824, saved readouts; 2026-11-11)
If yes, sampling explores productively and filtered self-training / RL has signal; if not, they cannot find it.
Element credit = the option's element centre lies inside the gold target box (ac_common.centre_in_box), the
contract's own credit rule (not InfiGUI's enlarged box), reported next to its own top-1 for reference.
    python topk_actest.py CODE_DIR TEST_COMPACT EVAL_JSONL"""
import json, sys, collections as C
sys.path.insert(0, sys.argv[1]); import ac_common as A
test = {(str(r["ep"]), str(r["step"])): r for r in map(json.loads, open(sys.argv[2]))}
ACT = {"click": "click", "long_press": "long_press", "type_text": "write", "scroll": "scroll", "open_app": "open_app",
       "navigate_back": "navigate_back", "navigate_home": "navigate_home", "wait": "wait"}
el = C.Counter(); act = C.Counter(); mass = []
for l in open(sys.argv[3]):
    e = json.loads(l); r = test.get((str(e["episode_id"]), str(e["step"])))
    if r is None: continue
    g = r["action"]["action_type"]
    pa = e["readouts"]["action"]["probabilities"]; order = sorted(pa, key=lambda k: -pa[k])
    if g in ACT:
        act["n"] += 1; rk = order.index(ACT[g]) + 1 if ACT[g] in order else 99
        for k in (1, 2, 3): act["top%d" % k] += rk <= k
    if g in ("click", "long_press") and r.get("target"):
        kept, idx = A.select_elements(r["elements"], int(r["W"]), int(r["H"]), "cap40_reading_noime")
        ok = {str(i + 1) for i, k in enumerate(kept) if A.centre_in_box(k["bbox"], r["target"])}
        if "target_el" not in e["readouts"]:          # no element list on this screen: counted, never credited
            el["n"] += 1; el["no target_el readout"] += 1; mass.append(0.0); continue
        p = e["readouts"]["target_el"]["probabilities"]; o = sorted(p, key=lambda k: -p[k])
        el["n"] += 1; el["gold reachable in list"] += bool(ok)
        for k in (1, 2, 3, 5): el["top%d" % k] += bool(ok & set(o[:k]))
        mass.append(sum(p[k] for k in ok))
print("action type, n=%d: top1 %.1f%%  top2 %.1f%%  top3 %.1f%%" % (act["n"], *(100 * act["top%d" % k] / act["n"] for k in (1, 2, 3))))
n = el["n"]; print("click target element, n=%d (gold reachable in the 40-list: %.1f%%): top1 %.1f%%  top2 %.1f%%  top3 %.1f%%  top5 %.1f%%" % (
    n, 100 * el["gold reachable in list"] / n, *(100 * el["top%d" % k] / n for k in (1, 2, 3, 5))))
print("rows without a target_el readout:", el["no target_el readout"])
print("mean probability on a correct element (= success rate of one sample at temperature 1): %.1f%%" % (100 * sum(mass) / len(mass)))
