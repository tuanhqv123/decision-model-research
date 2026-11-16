| Scenario | Set | Category | Success | Strict | Steps | Time (s) | LLM calls | Human requests |
|---|---|---|---|---|---|---|---|---|
| wifi_off | tuned | settings | 3/3 | 3/3 | 7.7 | 65 | 12.7 | 1.0 |
| wifi_on | tuned | settings | 3/3 | 3/3 | 5.0 | 31 | 6.0 | 0.0 |
| bluetooth_off | tuned | settings | 3/3 | 3/3 | 6.3 | 40 | 7.3 | 0.0 |
| dark_theme_on | tuned | settings | 3/3 | 3/3 | 5.0 | 29 | 6.0 | 0.0 |
| dnd_on | tuned | settings | 3/3 | 3/3 | 6.0 | 35 | 7.0 | 0.0 |
| screen_timeout | tuned | settings | 3/3 | 3/3 | 6.0 | 36 | 7.0 | 0.0 |
| note_create | tuned | notes | 3/3 | 3/3 | 16.0 | 91 | 11.7 | 5.0 |
| note_append | tuned | notes | 3/3 | 3/3 | 6.0 | 35 | 6.0 | 2.0 |
| note_delete | tuned | notes | 3/3 | 3/3 | 5.0 | 32 | 6.0 | 1.0 |
| contact_add | tuned | contacts | 3/3 | 3/3 | 11.0 | 61 | 8.7 | 0.0 |
| battery_level | tuned | question | 3/3 | 3/3 | 1.0 | 3 | 1.0 | 0.0 |
| android_version | tuned | question | 3/3 | 3/3 | 5.7 | 31 | 5.7 | 0.0 |
| airplane_on | held-out | settings | 3/3 | 3/3 | 5.3 | 30 | 6.3 | 0.0 |
| auto_rotate_on | held-out | settings | 3/3 | 3/3 | 5.0 | 29 | 6.0 | 0.0 |
| location_off | held-out | settings | 3/3 | 3/3 | 6.7 | 38 | 7.3 | 0.3 |
| font_larger | held-out | settings | 3/3 | 3/3 | 6.0 | 36 | 7.0 | 0.0 |
| note_rename | held-out | notes | 3/3 | 3/3 | 16.7 | 129 | 16.0 | 7.0 |
| contact_delete | held-out | contacts | 3/3 | 3/3 | 6.0 | 32 | 6.0 | 2.0 |
| note_question | held-out | question | 3/3 | 3/3 | 3.0 | 16 | 3.0 | 0.0 |
| device_name | held-out | question | 3/3 | 3/3 | 5.0 | 26 | 5.0 | 0.0 |

Held-out: completed 24/24 = 100.0% (95% CI 86.2–100.0%); strict 24/24 = 100.0% (95% CI 86.2–100.0%).
Tuned: completed 36/36 = 100.0% (95% CI 90.4–100.0%); strict 36/36 = 100.0% (95% CI 90.4–100.0%).
All: completed 60/60 = 100.0% (95% CI 94.0–100.0%); strict 60/60 = 100.0% (95% CI 94.0–100.0%).
Mean per run: 6.7 steps, 41 s, 7.1 LLM calls.
Mean latency per call (client side of the shared GPU / LLM endpoint): DMA-2B 142 ms, LLM 1286 ms.
