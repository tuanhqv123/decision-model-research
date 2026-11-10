# A1 — Can GUI-Odyssey supervise OUR failure? Measured, not assumed.

*Measured 2026-11-09 by downloading and parsing the real annotation files. Not from the dataset card.*
`all_anno.json` (33,691,272 bytes, sha `71e0e7e2d169c642e7a99c264d82ffecadf67889`) parsed in full:
**7,735 episodes, 118,791 steps, 0 unparseable.**

## VERDICT

**Partly — and the part it cannot do is the part we need least.**

The card is wrong about the schema, and the correction is *favourable*: there is no `TYPE` action. The
action is **`TEXT`**, and although it carries only the string, **99.3% of `TEXT` steps (12,314 of 12,405)
are immediately preceded by a `CLICK` that carries the field coordinates.** So GUI-Odyssey *does* supervise
typing, in click-then-type form, on the same 0–1000 grid we already use.

What it does **not** have at all: `open_app`, and a localised target for the `TEXT` step itself (confirmed
empty `sam2_bbox` in the newer version too). Both are derivable or already covered by our data.

**So the plan is not blocked.** My earlier reading of the card — that `TYPE` has no coordinates and therefore
the corpus cannot teach our one real failure — was **wrong**, because I read the card rather than the data.
The failure is still real, but the corpus does provide the supervision.

---

## 1. The actual schema (read, not quoted)

`all_anno.json` is a JSON **list** of 7,735 episode dicts. Each `steps` field is a **string** containing a
Python-literal list (`ast.literal_eval`, not `json.loads`) — a real ingestion trap.

Episode keys: `episode_id, device_name, h, w, category, app, meta_task, task, instruction, step_length, steps`

Step keys: **`step, screenshot, action, info, ps`** — five fields, not the eight the card's newer version has.

### Action vocabulary — measured over all 118,791 steps

| Action | Count | % | In the card? |
|---|---|---|---|
| `CLICK` | 86,847 | 73.11% | ✓ |
| **`TEXT`** | **12,405** | **10.44%** | ✗ card says `TYPE` |
| `SCROLL` | 11,312 | 9.52% | ✓ |
| `COMPLETE` | 7,486 | 6.30% | ✓ |
| `LONG_PRESS` | 492 | 0.41% | ✓ |
| **`INCOMPLETE`** | **249** | 0.21% | ✗ card says `IMPOSSIBLE` |
| `HOME`, `BACK` | **0** | 0% | ✗ card claims both exist |

Six actions, not eight. `HOME` and `BACK` **do not exist** in this corpus.

### What each `info` actually contains

| Action | `info` | Coordinates? |
|---|---|---|
| `CLICK` | `[[x, y], [x, y]]` | **yes**, 0–1000 normalised |
| `LONG_PRESS` | `[[x, y], [x, y]]` | **yes** |
| `SCROLL` | `[[x1,y1],[x2,y2]]` | **yes**, a swipe vector |
| `TEXT` | `"Todoist"` — **a bare string** | **no** |
| `COMPLETE` / `INCOMPLETE` | `""` | n/a |

## 2. The check that decides the plan: does `TEXT` have a target?

Measured over all 12,405 `TEXT` steps — the action immediately preceding:

| Preceding action | Count | % |
|---|---|---|
| **`CLICK`** | **12,314** | **99.27%** |
| `<episode start>` (no prior step) | 43 | 0.35% |
| `TEXT` (continuation of same field) | 37 | 0.30% |
| `SCROLL` | 11 | 0.09% |

**So the field *is* localised — by the click that precedes the typing.** A `TEXT` step means "the field that
was just clicked". That is exactly the `click → type_text` pattern our model must learn and fails to
sequence (the `ContactsAddContact` trace types once and never returns to the form).

In the **newer** version (`hflqf88888/GUIOdyssey`, sha `61632d0f...`) each step gained
`description`, `intention`, `low_level_instruction`, `context`, `sam2_bbox`. I checked the `TEXT` step there:
**`sam2_bbox` is `[]` — empty.** So even the improved release does not localise `TEXT` directly. But it adds
`low_level_instruction` ("Type 'Todoist' in the search bar.") and `context` (a running progress summary),
which is *better* progress supervision than the old version had. The click-precedes-type structure is
unchanged.

## 3. Corpus counts (what the training signal actually looks like)

| | |
|---|---|
| Episodes | **7,735** (card says 7,735; the paper says 8,334 — the card is right for this release) |
| Total steps | **118,791** |
| Median episode length | **14** steps (p90 = 24, max = 59) |
| Episodes ending in `COMPLETE` | **7,486 / 7,735 = 96.8%** |
| Episodes containing ≥1 `TEXT` | **6,707 / 7,735 = 86.7%** |
| Episodes with no `TEXT` at all | 1,028 |
| First action is `CLICK` | 7,468 (96.6%) |
| First action is `SCROLL` | 224 |
| First action is `TEXT` | 43 |

Screens: 1080×2400 (1,314), 1440×3120 (1,312), 1344×2992 (1,288), **2560×1600 landscape** (1,281),
720×1280 (1,278), 2208×1840 (1,252), 1840×2208 (10). **67.3% portrait** — so one-third is tablet/landscape,
which our model has never seen. That is a distribution shift to handle explicitly, not to ignore.

Categories: General_Tool 1,821 / Multi_Apps 1,616 / Information_Management 1,423 / Social_Sharing 1,316 /
Media_Entertainment 1,080 / Web_Shopping 479.

## 4. Mapping to our 8 readouts

| Our readout | GUI-Odyssey | Verdict |
|---|---|---|
| `click` | `CLICK` `[[x,y],[x,y]]`, 0–1000 | **full** |
| `long_press` | `LONG_PRESS` | **full** (but only 492 rows) |
| `scroll` | `SCROLL` `[[x1,y1],[x2,y2]]` | **derivable** — direction from the delta |
| `type_text` | `TEXT "<string>"` + preceding `CLICK` | **full, in two steps** — the field comes from the prior click; the string from the `TEXT` row |
| `status` | `COMPLETE` (7,486) / `INCOMPLETE` (249) | **full** |
| `open_app` | **absent** | **derive** from `CLICK` on a launcher icon after a `HOME`… but `HOME` is absent too, so **only partly derivable** |
| `navigate_back` | **absent** (`BACK` = 0) | **missing** — our data must cover it (it is 750/15,000 = 5% of cur15k) |
| `wait` | **absent** | **missing** (150/15,000 = 1% of ours; low impact) |

Derived scroll directions over all 11,312 `SCROLL` steps: up 7,791 / left 1,669 / right 1,165 / down 687.
Axis: vertical 8,478 (74.9%), horizontal 2,834 (25.1%).

## 5. What published users report

GUI-Odyssey (arXiv 2406.08451, ICCV 2025) and GUI-R1 (arXiv 2504.10458) are the known consumers.
**Neither publishes an end-to-end AndroidWorld/AndroidLab success rate for a ≤2B model trained on this corpus
alone** — GUI-Odyssey's own evaluations are on its own AITZ-style benchmark, and GUI-R1 is a 7B RL follow-up.
That matters for issue I3: this corpus's *measured* contribution to end-to-end success at our scale is
**unestablished** (see `A2-trajectory-sft-evidence.md`).

## 6. Alternatives, if we want a localised `TYPE` in one step

| Dataset | Size | Licence | Localises type? | Source |
|---|---|---|---|---|
| AndroidControl (already ours) | 15,000 rows in use, rebuilt from GCS | research | **yes** — 450 `type_text` rows carry a `target` bbox | our `compact_cur15k.jsonl` |
| GUI-Odyssey (old) | 7,735 eps / 118,791 steps | cc-by-4.0 | **via preceding click** (99.3%) | `OpenGVLab/GUI-Odyssey` |
| GUI-Odyssey (new) | same corpus, richer fields | cc-by-4.0 | no (`sam2_bbox` empty for TEXT) but adds `low_level_instruction` + `context` | `hflqf88888/GUIOdyssey` |
| AMEX | 1.2M Android element annotations | CC-BY-4.0 | element annotations, not action trajectories | — |
| AndroidLab | 138 tasks | research | harness, not a trajectory corpus | `xuyifan0731/mobilerl-androidlab-eval` |

## 7. What I could not verify

- Whether `TEXT` in the **new** release ever sets `sam2_bbox` non-empty (I sampled one episode; one `TEXT`
  step; it was `[]`). I did not scan all 7,735 new-version annotation files — that is 7,735 HTTP requests.
- The screenshots (`screenshots/screenshots.z01…z08` + `.zip`) were **not downloaded**. Their availability
  and total size are unverified.
- The 8,334 vs 7,735 discrepancy: the arXiv paper says 8,334; the card and the actual `all_anno.json` say
  **7,735**. I trust the file. The 599-episode gap is unexplained.
- GUI-R1's exact training mix (whether it uses GUI-Odyssey only or mixed) — I did not fetch the full paper.

## 8. Consequences for the plan

1. **Correct the issue document.** I2 said "`TYPE` has empty `info`, so this corpus cannot teach field
   selection." That is **wrong** — the field is carried by the preceding `CLICK` on 99.3% of rows. The correct
   statement: *the corpus supplies typing supervision in two-step form, which means the training example must
   be constructed as a (click, type) pair rather than a single row.*
2. **Two readouts need our own data**: `navigate_back` and `wait` are absent. They are 6% of cur15k. Keep
   them from AndroidControl; do not expect GUI-Odyssey to teach them.
3. **`open_app` needs derivation** and its correctness is unverified — the corpus has no home/launcher action,
   so a "launch app X" prefix has to be synthesised from the first `CLICK`.
4. **One-third of screens are non-portrait.** Any recipe must decide whether to resample or accept the shift.
5. **The 86.7% typing rate is itself the strongest argument for this corpus.** 86.7% of its episodes involve
   text entry, versus 19/104 in our AndroidWorld run. That is precisely the imbalance we need to correct
   (issue I4, mechanism M1).
