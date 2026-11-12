# Error analysis, v3h (v3 + harness V8, seed 30) — first wrong step per failed episode
Method: skill `error-analysis` (read traces, note the FIRST thing that went wrong, categories emerge after ~30).
| # | task | first wrong step | observation |
|---|---|---|---|
| 0 | AudioRecorderRecordAudioWithFileName | 1-5 | opened settings-like flow (Get started/Naming/Apply), never pressed record, then status |
| 1 | BrowserDraw | 5 | Chrome first-run "Use without an account": click → no change, then waits 12x; flow never reaches the page |
| 2 | BrowserMaze | 3 | same Chrome first-run dialog, click no change, waits |
| 3 | BrowserMultiply | 4 | in Chrome first-run picked "Add account" → Google sign-in, types into email field 15x (no change) |
| 4 | CameraTakePhoto | 2 | shutter at step 1 changed screen (photo taken?), keeps clicking shutter 8x, never stops |
| 5 | CameraTakeVideo | 1 | shutter (photo mode, not video) then repeats |
| 6 | ClockTimerEntry | 3 | goal 16 min 35 s: keys '1' then '0','0',... — wrong digit key chosen (should be 6,3,5) |
| 7 | ContactsNewContactDraft | 5-6 | typed label "Work" into Company field; never stopped (budget) |
| 8 | ExpenseAddMultiple | 5 | category click "Social" → no change (selector not opened); saved 1 of 3 then status |
| 9 | ExpenseAddMultipleFromGallery | 1 | wrong plan: creates a gallery folder named "pro expense" |
| 10 | ExpenseAddMultipleFromMarkor | 0 | opened pro expense instead of Markor (needs source file first) |
| 11 | ExpenseAddSingle | 5 | skipped category "Health Care" (clicked Note), saved, status |
| 12 | ExpenseDeleteDuplicates | 2 | deletes items that are not the duplicates (Flight Tickets, Wellness) then status |
| 13 | ExpenseDeleteDuplicates2 | 1 | deletes arbitrary items (Membership, Workshops, Dining) — no duplicate identification |
| 14 | ExpenseDeleteMultiple2 | 5 | 2nd delete hits a different expense ('190.95' row), status after 2 of 3 |
| 15 | FilesDeleteFile | 7 | opened distractor `edited_q2a8_fancy_banana.mp3` (target `q2a8_fancy_banana.mp3`); "open with" sheet, "Just once" disabled → no change x14 |
| 16 | FilesMoveFile | 1 | wrong plan: More options → Select all in root; then waits |
| 17 | MarkorAddNoteHeader | 3 | type_text REPLACED the note (clear_text) instead of adding a header; then Save → no change x8 |
| 18 | MarkorChangeNoteContent | 4 | content replaced correctly; Save → no change (Markor saves with no UI change) x8; rename never done |
| 19 | MarkorCreateNote | 2 | name typed without ".md"; then cycles template dropdown "Empty file", picks a reference template; Save loop |
| 20 | MarkorCreateNoteAndSms | 6 | note created + text typed; Save → no change x12; never goes to the SMS part |
| 21 | MarkorCreateNoteFromClipboard | 7 | created file, never pasted; Save → no change, status |
| 22 | MarkorDeleteAllNotes | 1 | opens a note and uses editor "Delete lines" instead of deleting files in the list; status |
| 23 | MarkorDeleteNewestNote | 1 | same: opens note, "Delete lines", status |
| 24 | MarkorDeleteNote | 1 | same: opens note, "Delete lines", status |
| 25 | MarkorEditNote | 3 | "add to top": type_text replaced the whole note with "Hello, World" (also dropped "!"); Save, status |
| 26 | MarkorMergeNotes | 2 | (unscored) creates note with random name, picks template; wrong plan |
| 27 | MarkorMoveNote | 3 | opened the note then More options → Share → Nearby: no "move" path; Continue → no change loop |
| 28 | MarkorTranscribeReceipt | 4 | (unscored) picked FOLDER type → made a folder "receipt.md" |
| 29 | MarkorTranscribeVideo | 1 | VLC: waits on "Loading"/library x19, never opens Download |
| 30 | NotesIsTodo | 2 | IR task: on the note, status (v3 has no answer action) |
| 31 | NotesMeetingAttendeeCount | 4 | IR: reached the note, the answer element "25 individuals were present" was chosen, but status (no answer) |
| 32 | NotesRecipeIngredientCount | 4 | IR: reached the recipe; element "3/4 cup" chosen; status (no answer) |
| 33 | NotesTodoItemCount | 1 | IR: counting not attempted; status |
| 34 | OsmAndMarker | 4 | searched + selected place; tapped "Marker" (a tab?) then status — marker not added |
| 35 | OsmAndTrack | 6 | route search: clicks "Send" alternating screens for 110 steps |
| 36 | RecipeAddMultipleRecipes | ? | (long goal; see trace) |
| 37 | RecipeAddMultipleRecipesFromImage | 1 | wrong plan: creates gallery folder |
| 38 | RecipeAddMultipleRecipesFromMarkor | 2 | opens recipes.txt, Save → no change, status (never goes to Broccoli) |
| 39 | RecipeAddMultipleRecipesFromMarkor2 | 2 | opens recipes.txt, overwrites it with "Broccoli", status |
| 40 | RecipeAddSingleRecipe | 4 | Title ok; click Description → no change ... (budget) |
| 41 | RecipeDeleteDuplicateRecipes2 | 1 | deletes a non-duplicate (first recipe), status |
| 42 | RecipeDeleteDuplicateRecipes3 | 1 | same |
| 43 | RecipeDeleteMultipleRecipes | 5 | deletes 1 of 3, status |
| 44 | RecipeDeleteMultipleRecipesWithConstraint | 6 | scrolls to list end, scroll → no change x34 (needs to open recipes to read directions) |
| 45 | RecipeDeleteMultipleRecipesWithNoise | 2 | target "Grilled Cheese..." visible at step 2 but keeps scrolling; scroll → no change x28 |
| 46 | RetroCreatePlaylist | 7 | playlist created; then "(no label)" row-menu buttons pick songs it cannot identify (unlabelled); wrong songs; status |
| 47 | RetroPlayingQueue | 2 | wrong plan: creates a playlist named after the first song; adds unlabelled rows to queue |
| 48 | RetroPlaylistDuration | 7 | creates empty playlist, status |
| 49 | RetroSavePlaylist | 7 | same as 46 (unlabelled song menus), then Import playlist |
| 50 | SaveCopyOfReceiptTaskEval | 4 | wrong plan: creates folder "Download" instead of copy |
| 51 | SimpleCalendarAddOneEvent | 11-12 | date picker: day scrolled 15→17→…→**29 reached at step 11**, keeps scrolling past it; never presses OK |
| 52 | SimpleCalendarAddOneEventInTwoWeeks | 7 | opens an existing event "Meeting with HR", click → no change x25 |
| 53 | SimpleCalendarAddOneEventRelativeDay | 7 | "(no label)" FAB click → no change x27 |
| 54 | SimpleCalendarAddOneEventTomorrow | 4 | "(no label)" click → no change loops; search detour |
| 55 | SimpleCalendarAddRepeatingEvent | 4 | searches for the event instead of creating; "(no label)" no-change loops |
| 56 | SimpleCalendarAnyEventsOnDate | 3 | IR: date picker scrolling 13→15→…→26 (budget 10) |
| 57 | SimpleCalendarDeleteEvents | 5 | Go to date picker: scroll "OK" button x9 (scroll anchored on OK) |
| 58 | SimpleCalendarDeleteEventsOnRelativeDay | 3 | Go to date → OK without choosing date; status |
| 59 | SimpleCalendarDeleteOneEvent | 6 | date picker: scroll anchored on "OK" x6 |
| 60 | SimpleCalendarEventOnDateAtTime | 4 | IR: date picker scrolled 2 days per step, budget runs out |
