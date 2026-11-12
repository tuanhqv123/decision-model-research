"""First-error category per failed v3h episode (index = show_trace.py --fail-only order). Read by hand 2026-11-12."""
CAT = {
 "APP_PROCEDURE": "does not know how this app does the task (wrong path/menu/app), first wrong step is a plan step",
 "TEXT_ENTRY": "typing semantics: replaced instead of inserting, dropped file extension, wrong field",
 "WIDGET": "special widget: date/number picker, keypad digits, selector, unlabelled '(no label)' control",
 "MULTI_ITEM_STOP": "multi-item goal: does item 1 (of N) correctly, then status",
 "IR_ANSWER": "information-retrieval task: needs an answer (v3 has no answer action / reads wrong info)",
 "WRONG_INSTANCE": "picks the wrong one among similar items (distractor file, duplicates, other row)",
 "NO_EFFECT_LOOP": "an action that is correct or harmless shows no UI change; model repeats it until budget",
 "ENV": "environment: Chrome first-run dialog, no SIM (SMS)",
}
L = {0:"APP_PROCEDURE",1:"ENV",2:"ENV",3:"ENV",4:"NO_EFFECT_LOOP",5:"APP_PROCEDURE",6:"WIDGET",7:"TEXT_ENTRY",8:"WIDGET",
 9:"APP_PROCEDURE",10:"APP_PROCEDURE",11:"WIDGET",12:"WRONG_INSTANCE",13:"WRONG_INSTANCE",14:"WRONG_INSTANCE",15:"WRONG_INSTANCE",
 16:"APP_PROCEDURE",17:"TEXT_ENTRY",18:"NO_EFFECT_LOOP",19:"TEXT_ENTRY",20:"NO_EFFECT_LOOP",21:"APP_PROCEDURE",22:"APP_PROCEDURE",
 23:"APP_PROCEDURE",24:"APP_PROCEDURE",25:"TEXT_ENTRY",26:"ENV",27:"APP_PROCEDURE",28:"ENV",29:"APP_PROCEDURE",30:"IR_ANSWER",
 31:"IR_ANSWER",32:"IR_ANSWER",33:"IR_ANSWER",34:"APP_PROCEDURE",35:"APP_PROCEDURE",36:"MULTI_ITEM_STOP",37:"APP_PROCEDURE",
 38:"APP_PROCEDURE",39:"TEXT_ENTRY",40:"WIDGET",41:"WRONG_INSTANCE",42:"WRONG_INSTANCE",43:"MULTI_ITEM_STOP",44:"APP_PROCEDURE",
 45:"NO_EFFECT_LOOP",46:"WIDGET",47:"APP_PROCEDURE",48:"MULTI_ITEM_STOP",49:"WIDGET",50:"APP_PROCEDURE",51:"WIDGET",52:"APP_PROCEDURE",
 53:"WIDGET",54:"WIDGET",55:"APP_PROCEDURE",56:"IR_ANSWER",57:"WIDGET",58:"WIDGET",59:"WIDGET",60:"IR_ANSWER",61:"IR_ANSWER",
 62:"IR_ANSWER",63:"IR_ANSWER",64:"IR_ANSWER",65:"IR_ANSWER",66:"IR_ANSWER",67:"IR_ANSWER",68:"APP_PROCEDURE",69:"ENV",70:"ENV",
 71:"ENV",72:"ENV",73:"ENV",74:"ENV",75:"IR_ANSWER",76:"IR_ANSWER",77:"IR_ANSWER",78:"IR_ANSWER",79:"IR_ANSWER",80:"IR_ANSWER",
 81:"WIDGET",82:"APP_PROCEDURE",83:"APP_PROCEDURE",84:"APP_PROCEDURE",85:"APP_PROCEDURE",86:"APP_PROCEDURE",87:"APP_PROCEDURE",
 88:"APP_PROCEDURE",89:"IR_ANSWER",90:"IR_ANSWER",91:"IR_ANSWER",92:"IR_ANSWER",93:"IR_ANSWER",94:"IR_ANSWER",95:"APP_PROCEDURE",
 96:"APP_PROCEDURE",97:"APP_PROCEDURE",98:"APP_PROCEDURE"}
# Unscored episodes (checker raised / never ran) are infrastructure outcomes, not model failures (AGENTS.md M5):
# MarkorMergeNotes (#26), MarkorTranscribeReceipt (#28), SimpleSmsReplyMostRecent (#70, 0 steps).
UNSCORED = {26, 28, 70}

if __name__ == "__main__":
    import collections
    scored = {k: v for k, v in L.items() if k not in UNSCORED}
    c = collections.Counter(scored.values()); n = len(scored)
    print(f"{n} scored failed episodes (v3h, seed 30; {len(L)} listed, {len(UNSCORED)} unscored excluded)")
    for k, v in c.most_common(): print(f"  {k:16s} {v:3d}/{n}  {v/n:5.1%}  {CAT[k]}")
