# AndroidWorld protocol — what every reported number must carry

**Protocol string:** `aw-main-v1`. Any number that does not say this string comes from a different,
non-comparable setup. AndroidWorld numbers are not comparable across papers (GUI-Libra-32B 34.8 vs
UI-Venus-2-9B 57.8 under the same benchmark name), so we report our own controls and our own protocol,
or we report nothing.

## Required beside every number

| Field | Why it is mandatory |
|---|---|
| `protocol` | identifies the harness revision that produced it |
| `agent`, `checkpoint` | a number without a checkpoint cannot be reproduced |
| `n` and denominator | a rate without its denominator is not a result |
| `SR_all` **and** `SR_reachable` | the checkpoint emits 8 of AndroidWorld's 13 action types; without the split, "press Enter" reads as "bad at GUI" |
| `coverage` per task | emittable gold action types / all gold action types; `null` when the suite cannot report them |
| crash rate | a crashed episode is not a failure of the model |
| timeout rate | an episode that never terminated is not a wrong answer |
| invalid-action rate | the model emitting an action AW cannot represent |
| `grounding_error_proxy` | **proxy only** — online there is no gold coordinate; never present it as the AC-v2 grounding number |
| model p50/p95 **and** env ms | the environment dominates end-to-end; one number would hide that |
| `tokens_generated` | the single-pass claim is only meaningful if this is 0 |

## Baselines that must be run on the identical instance tuples

| Agent | Question it answers |
|---|---|
| `wait` | does the task complete by itself? |
| `random` | what does a policy with no model knowledge score? |
| `m3a` | AndroidWorld's own reference agent |
| `d1_zero_shot` | how much did fine-tuning actually contribute? |
| `d1_sft` | the checkpoint we report |

Pairing is verified by reading `pairing_key` back out of each episode header, not asserted in prose.

## Gates

No benchmark number is produced before **Gate A** (environment correctness) and **Gate B** (benchmark
correctness) pass. A full run on a broken harness is a retraction, and this project has already had two.

- **Gate A:** liveness passes all four layers; every emittable action round-trips on a live device;
  `type_text` types spaces and non-ASCII correctly; `open_app` succeeds for all 24 canonical apps.
- **Gate B:** a 5-task smoke runs twice with identical results at the same seed; every action type
  appears at least once; the trajectory replays from JSONL.
- **Gate C:** a 15–20 task pilot produces an error taxonomy with per-bucket counts, stable across two
  seeds, before any 116-task run.

## What "not a result" means

The 6-task `1/6` vs `0/6` run (2026-11-02) is a smoke signal, not a benchmark result: six tasks, a
pixels-only HTTP server that cannot execute `type_text`, an `app_name` fallback, and a Wilson band of
3–56%. **The paper's §5.7 now reports the 8-task protocol: SR `1/8` in both conditions, Wilson
2.2–47.1%, with the Clock-checker crash root cause.** The `1/6` pair may be quoted as history only.
