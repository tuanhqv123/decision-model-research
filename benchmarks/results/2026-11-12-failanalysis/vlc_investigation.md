# VLC "Loading" in AndroidWorld (investigated on an emulator, 2026-11-12)

- **Observation (traces):** in all 8 VLC episodes across four runs, VLC's Playlists/Video tabs show "Loading" for the whole episode.
- **Reproduction:** instantiate `VlcCreatePlaylist` with the official seed, run its `initialize_task`, launch VLC. The
  main screen shows "Loading" at +5 s, +15 s and +30 s, and in the Video tab. The task's two video files were written to
  `/storage/emulated/0/VLCVideos`.
- **Log (logcat):** the media library initialises ("Successfully initialized") and starts discovery. It then reports
  `Creation of a folder failed: Request [INSERT INTO Folder(...)] aborted due to constraint violation (UNIQUE constraint failed: Folder.path, Folder.device_id). Assuming it was banned`.
  The upstream task setup deletes all rows of the Playlist, Media and PlaylistMediaRelation tables with `sqlite3` as root
  (`_clear_playlist_dbs`), while the Folder table keeps its rows.
- **Same after upstream `VlcApp.setup`:** clear data, Skip; it stopped at "GRANT PERMISSION", already granted.
- **Same on every emulator:** same APK as the official image, so this is treated as a benchmark condition, not an
  environment defect of ours.
- **Task still solvable:** the Browse tab lists Internal memory and its folders (file browsing does not use the media library).
- **Status:** cause = inferred from the log line and the setup code. The fix inside VLC's database is not attempted
  (it would modify app state beyond the task's own setup).
