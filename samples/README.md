# samples/

Short **royalty-free** demo tracks (original Relentless music) so NoteSight runs
out of the box. Point the CLI at any of them:

```bash
notesight samples/Neon.ogg -d medium -o out/
```

## Tracks

| file | vibe | lyrics |
|---|---|---|
| `Brain Rot (Only In Ohio).ogg` | vocal | ✅ `.lrc` included |
| `Boogie.ogg` `Funky.ogg` `Horizon.ogg` `Midnight.ogg` `Neon.ogg` `Pulse.ogg` `Swingin'.ogg` | EDM, instrumental | — |

**Brain Rot** ships with a sidecar `Brain Rot (Only In Ohio).lrc` to demonstrate
**lyric-aligned pattern reuse**: the charter finds where the chorus (the words)
comes back and mirrors the pattern there. The CLI auto-detects the `.lrc`; add
`--no-lrc` to hear the difference:

```bash
notesight "samples/Brain Rot (Only In Ohio).ogg" -d hard -o out/           # uses lyrics
notesight "samples/Brain Rot (Only In Ohio).ogg" -d hard --no-lrc -o out/  # acoustic only
```

The instrumental EDM tracks use acoustic self-similarity for structure.

Only original or royalty-free music belongs here — never copyrighted songs.
