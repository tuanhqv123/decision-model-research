| Scenario | Category | Result | Steps | Time (s) | LLM calls | Recoveries | Human requests |
|---|---|---|---|---|---|---|---|
| wifi_off | settings | ✓ | 5 | 32 | 6 | 0 | 0 |
| wifi_on | settings | ✓ | 5 | 31 | 6 | 0 | 0 |
| bluetooth_off | settings | ✓ | 16 | 95 | 17 | 0 | 0 |
| dark_theme_on | settings | ✓ | 5 | 29 | 6 | 0 | 0 |
| dnd_on | settings | ✓ | 26 | 399 | 89 | 0 | 5 |
| screen_timeout | settings | ✓ | 6 | 37 | 7 | 0 | 0 |
| note_create | notes | ✓ | 30 | 214 | 28 | 0 | 13 |
| note_append | notes | ✓ | 6 | 37 | 6 | 0 | 2 |
| note_delete | notes | ✓ | 5 | 31 | 6 | 0 | 1 |
| contact_add | contacts | ✓ | 10 | 61 | 8 | 0 | 0 |
| battery_level | question | ✓ | 1 | 3 | 1 | 0 | 0 |
| android_version | question | ✓ | 7 | 36 | 7 | 0 | 0 |

Completion rate: **12/12 = 100.0%**.
Mean per task: 10.2 steps, 84 s, 15.6 LLM calls.
Mean latency: DMA-2B 270 ms per decision, LLM 1343 ms per call.
