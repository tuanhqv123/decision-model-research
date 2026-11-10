"""Independent post-checks on the written v2 files: v1 fields unchanged, schema, extra diagnostics.

Usage: python post_check.py V1_TRAIN V2_TRAIN [V1_VAL V2_VAL]   (streams both files in lockstep)
"""
import json, sys, collections
NEW = ["elements", "target_el_idx", "target_el_exact", "target_el_hits_gold", "target_el_centre_idx", "effect",
       "effect_app", "pixel_change", "local_goal", "local_done", "n_steps", "elements_source"]
SP = ("click", "long_press", "type_text")
for v1p, v2p in zip(sys.argv[1::2], sys.argv[2::2]):
    bad = collections.Counter()
    diag = collections.Counter()
    sysui = []
    n = 0
    eps_steps = collections.defaultdict(set)
    with open(v1p) as f1, open(v2p) as f2:
        for l1, l2 in zip(f1, f2):
            r1, r2 = json.loads(l1), json.loads(l2)
            n += 1
            if {k: r2.get(k) for k in r1} != r1:
                bad["v1_field_changed"] += 1
            if list(r2)[: len(r1)] != list(r1):
                bad["v1_key_order"] += 1
            if set(r2) - set(r1) != set(NEW):
                bad["new_keys"] += 1
            els = r2["elements"]
            i = r2["target_el_idx"]
            if i is not None and not (0 <= i < len(els)):
                bad["idx_range"] += 1
            at = r2["action"]["action_type"]
            st = at == "status"
            if st != (r2["step"] == r2["n_steps"]):
                bad["status_vs_n_steps"] += 1
            if st and any(r2[k] is not None for k in ("effect", "effect_app", "pixel_change", "local_goal", "local_done")):
                bad["status_nonnull"] += 1
            if not st and any(r2[k] is None for k in ("effect", "effect_app", "pixel_change", "local_goal")):
                bad["nonstatus_null"] += 1
            if at in SP:
                t = r2["target"]
                if (t["x_max"] - t["x_min"]) * (t["y_max"] - t["y_min"]) <= 0:
                    diag["spatial_zero_area_gold"] += 1
                has = any(e["bbox_pixels"] == {"x_min": t["x_min"], "y_min": t["y_min"], "x_max": t["x_max"], "y_max": t["y_max"]} for e in els)
                if has and not r2["target_el_exact"]:
                    diag[f"aw_has_gold_but_smallest_is_other:{at}"] += 1
                if i is None:
                    diag[f"no_target_el:{at}"] += 1
                    if (t["x_max"] - t["x_min"]) * (t["y_max"] - t["y_min"]) > 0.5 * r2["W"] * r2["H"]:
                        diag["no_target_el_and_gold_gt_half_screen"] += 1
                if t and (t["x_max"] - t["x_min"]) * (t["y_max"] - t["y_min"]) > 0.5 * r2["W"] * r2["H"]:
                    diag["spatial_gold_gt_half_screen"] += 1
            sysui.append(sum(1 for e in els if e.get("package_name") == "com.android.systemui"))
            diag["el_text_or_desc"] += sum(1 for e in els if e["text"] or e["content_description"])
            diag["el_total"] += len(els)
            diag["el_clickable"] += sum(1 for e in els if e["is_clickable"])
            diag["el_checkable"] += sum(1 for e in els if e["is_checkable"])
            diag["el_zero_area"] += sum(1 for e in els if (e["bbox_pixels"]["x_max"] - e["bbox_pixels"]["x_min"]) * (e["bbox_pixels"]["y_max"] - e["bbox_pixels"]["y_min"]) <= 0)
            eps_steps[r2["ep"]].add(r2["step"])
        assert f1.readline() == "" and f2.readline() == "", "line counts differ"
    s = sorted(sysui)
    print(v2p, "rows", n, "bad", dict(bad))
    print("  diag", dict(diag))
    print("  systemui elements per screen p50", s[len(s) // 2], "p95", s[int(0.95 * (len(s) - 1))], "mean", round(sum(s) / len(s), 2))
    print("  rows per episode", collections.Counter(len(v) for v in eps_steps.values()).most_common(6))
