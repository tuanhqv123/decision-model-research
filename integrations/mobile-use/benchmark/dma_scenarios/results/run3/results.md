| Scenario | Category | Result | Steps | Time (s) | LLM calls | Recoveries | Human requests |
|---|---|---|---|---|---|---|---|
| wifi_off | settings | ✓ | 30 | 311 | 44 | 0 | 5 |
| wifi_on | settings | ✓ | 21 | 176 | 30 | 0 | 2 |
| bluetooth_off | settings | ✓ | 12 | 74 | 11 | 0 | 3 |
| dark_theme_on | settings | ✓ | 18 | 120 | 19 | 0 | 1 |
| dnd_on | settings | ✓ | 13 | 84 | 12 | 0 | 0 |
| screen_timeout | settings | ✓ | 6 | 39 | 7 | 0 | 0 |
| note_create | notes | ✓ | 27 | 229 | 39 | 0 | 13 |
| note_append | notes | ✓ | 30 | 260 | 30 | 0 | 21 |
| note_delete | notes | ✓ | 5 | 32 | 6 | 0 | 1 |
| contact_add | contacts | ✓ | 10 | 72 | 11 | 0 | 0 |
| battery_level | question | ✓ | 1 | 3 | 1 | 0 | 0 |
| android_version | question | ✓ | 5 | 31 | 5 | 0 | 0 |

Completion rate: **12/12 = 100.0%**.
Mean per task: 14.8 steps, 119 s, 17.9 LLM calls.
Mean latency: DMA-2B 278 ms per decision, LLM 1641 ms per call.
