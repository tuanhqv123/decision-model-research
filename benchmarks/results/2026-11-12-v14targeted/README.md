# v4 + harness D (V14) and environment repairs, targeted re-run (2026-11-12)

The same v4 checkpoint (ckpt 2482), seed 30, pass@1. Only the 39 tasks whose outcome harness D (numeric answer format,
Back closes the keyboard after typing) or the environment repairs (Markor first-run screens, OsmAnd marker database)
could change were re-run; the other 77 keep their harness C result
(`../2026-11-12-maskenv/aw116_v4g_s0{,_b,_c}`). Slot 7 holds two tasks moved from slot 5.

| | value |
|---|---|
| tasks re-run and scored | 37 of 39 |
| not re-run | RetroSavePlaylist (every attempt cut after about 3 minutes / 16 steps, 20 attempts), SimpleSmsSend (interrupted when the model server was stopped for training); both keep their harness C result |
| changed outcomes (C -> D) | ContactsAddContact 0 -> 1, NotesMeetingAttendeeCount 0 -> 1, SimpleCalendarLocationOfEvent 1 -> 0, SimpleSmsReply 1 -> 0; OsmAndMarker now scored (setup error -> 0) |
| paired | 2 wins / 2 losses |
| merged 116-task result | score 25.0 / 116 scored = 21.6% (24 full successes) |

Reading: the repairs make OsmAndMarker scorable, but harness D does not change the score (2 wins, 2 losses, within
replay noise: 6 of 20 same-host replays diverged in an earlier check).

Open: episodes of RetroSavePlaylist and SimpleCalendarAddRepeatingEvent were abandoned about every 3 minutes
(17 steps or fewer) on two different emulators; the ninth attempt of SimpleCalendarAddRepeatingEvent ran to the
34-step budget. The component that cuts them has not been identified.

Recompute: `python3 benchmarks/aw/aw_report.py` on the run directories, or the snippet in the commit message.
