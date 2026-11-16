| Scenario | Category | Result | Steps | Time (s) | LLM calls | Recoveries | Human requests |
|---|---|---|---|---|---|---|---|
| wifi_off | settings | ✓ | 10 | 123 | 9 | 0 | 1 |
| wifi_on | settings | ✓ | 5 | 61 | 6 | 0 | 0 |
| bluetooth_off | settings | ✓ | 6 | 72 | 7 | 0 | 0 |
| dark_theme_on | settings | ✓ | 5 | 59 | 6 | 0 | 0 |
| dnd_on | settings | ✓ | 8 | 125 | 14 | 0 | 0 |
| screen_timeout | settings | ✓ | 5 | 58 | 6 | 0 | 0 |
| note_create | notes | ✓ | 30 | 387 | 31 | 0 | 14 |
| note_append | notes | ✓ | 6 | 83 | 7 | 0 | 2 |
| note_delete | notes | ✓ | 5 | 64 | 6 | 0 | 1 |
| contact_add | contacts | ✓ | 12 | 137 | 9 | 1 | 0 |
| battery_level | question | ✓ | 2 | 18 | 2 | 0 | 0 |
| android_version | question | ✗ | 13 | 918 | 3 | 0 | 0 |

Completion rate: **11/12 = 91.7%**.
Mean per task: 8.9 steps, 175 s, 8.8 LLM calls.
Mean latency: DMA-2B 270 ms per decision, LLM 1360 ms per call.
