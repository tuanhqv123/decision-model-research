| Scenario | Set | Category | Success | Strict | Steps | Time (s) | LLM calls | Human requests |
|---|---|---|---|---|---|---|---|---|
| wifi_off | tuned | settings | 3/3 | 3/3 | 5.0 | 115 | 6.0 | 0.0 |
| wifi_on | tuned | settings | 3/3 | 3/3 | 5.0 | 117 | 6.0 | 0.0 |
| bluetooth_off | tuned | settings | 3/3 | 3/3 | 6.0 | 134 | 7.0 | 0.0 |
| dark_theme_on | tuned | settings | 3/3 | 3/3 | 5.0 | 93 | 6.0 | 0.0 |
| dnd_on | tuned | settings | 3/3 | 3/3 | 6.0 | 123 | 7.0 | 0.0 |
| screen_timeout | tuned | settings | 3/3 | 3/3 | 6.0 | 124 | 7.7 | 0.0 |
| note_create | tuned | notes | 3/3 | 3/3 | 13.0 | 273 | 12.3 | 5.3 |
| note_append | tuned | notes | 3/3 | 3/3 | 8.3 | 168 | 8.3 | 4.3 |
| note_delete | tuned | notes | 3/3 | 3/3 | 5.0 | 125 | 6.0 | 1.0 |
| contact_add | tuned | contacts | 3/3 | 3/3 | 10.3 | 210 | 8.0 | 0.0 |
| battery_level | tuned | question | 3/3 | 3/3 | 1.0 | 3 | 1.0 | 0.0 |
| android_version | tuned | question | 3/3 | 3/3 | 5.7 | 74 | 5.7 | 0.0 |
| airplane_on | held-out | settings | 3/3 | 3/3 | 5.0 | 97 | 6.0 | 0.0 |
| auto_rotate_on | held-out | settings | 3/3 | 3/3 | 5.7 | 84 | 6.7 | 0.0 |
| location_off | held-out | settings | 3/3 | 3/3 | 7.3 | 119 | 8.0 | 0.3 |
| font_larger | held-out | settings | 3/3 | 3/3 | 6.0 | 124 | 7.0 | 0.0 |
| note_rename | held-out | notes | 3/3 | 3/3 | 12.0 | 245 | 9.3 | 3.7 |
| contact_delete | held-out | contacts | 3/3 | 3/3 | 6.0 | 122 | 6.0 | 2.0 |
| note_question | held-out | question | 3/3 | 3/3 | 3.0 | 60 | 3.0 | 0.0 |
| device_name | held-out | question | 3/3 | 3/3 | 10.7 | 151 | 11.0 | 0.7 |

Held-out: completed 24/24 = 100.0% (95% CI 86.2–100.0%); strict 24/24 = 100.0% (95% CI 86.2–100.0%).
Tuned: completed 36/36 = 100.0% (95% CI 90.4–100.0%); strict 36/36 = 100.0% (95% CI 90.4–100.0%).
All: completed 60/60 = 100.0% (95% CI 94.0–100.0%); strict 60/60 = 100.0% (95% CI 94.0–100.0%).
Mean per run: 6.6 steps, 128 s, 6.9 LLM calls.
Mean latency per call (client side of the shared GPU / LLM endpoint): DMA-2B 10883 ms, LLM 1615 ms.
