# eval/ — accuracy harness

Score NoteSight's output against **hand-authored** reference charts, to check the
generator lands notes where human charters do. You supply the reference charts
(e.g. a StepMania `Songs/` folder you own) — none are shipped.

- `smparse.py` — a minimal `.sm` reader: parse a StepMania chart into ground-truth
  note times (seconds). Used by the scorers below.
- `compare.py` — score **one** song: BPM/offset error, onset coverage, note F-score.
  ```bash
  python eval/compare.py path/to/Song.sm
  ```
- `batch.py` — run a whole `Songs/` folder and print per-song rows + aggregate
  medians.
  ```bash
  python eval/batch.py "path/to/StepMania/Songs/SomePack"
  ```

Run from the repo root (the `notesight` package resolves via the editable install;
`compare`/`batch` import `smparse` from this folder).
