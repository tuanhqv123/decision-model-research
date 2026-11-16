| Scenario | Category | Result | Steps | Time (s) | LLM calls | Recoveries | Human requests |
|---|---|---|---|---|---|---|---|
| wifi_off | settings | ✓ | 5 | 76 | 6 | 0 | 0 |
| wifi_on | settings | ✓ | 5 | 60 | 12 | 0 | 0 |
| bluetooth_off | settings | ✓ | 6 | 73 | 19 | 0 | 0 |
| dark_theme_on | settings | ✓ | 5 | 61 | 25 | 0 | 0 |
| dnd_on | settings | ✗ | 1 | 21 | 27 | 0 | 0 |
| screen_timeout | settings | ✓ | 15 | 281 | 61 | 0 | 1 |
| note_create | notes | ✓ | 17 | 282 | 94 | 0 | 5 |
| note_append | notes | ✓ | 6 | 77 | 101 | 0 | 2 |
| note_delete | notes | ✓ | 5 | 68 | 107 | 0 | 1 |
| contact_add | contacts | ✓ | 12 | 190 | 122 | 0 | 0 |
| battery_level | question | ✓ | 2 | 24 | 124 | 0 | 0 |
| android_version | question | ✗ | 4 | 211 | 127 | 0 | 0 |

Completion rate: **10/12 = 83.3%**.
Mean per task: 6.9 steps, 119 s, 68.8 LLM calls.
Mean latency: DMA-2B 272 ms per decision, LLM 1608 ms per call.
