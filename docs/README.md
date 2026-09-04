# ES-FA Notes and Reports

This folder is for readable project notes: setup, architecture, progress,
checklists, and LaTeX documents. It is intentionally separate from generated
tables and plots, which go under `results/`.

## Files

- `first_time_user_setup_guide.tex`  
  Step-by-step setup guide, starting from the KV260 SD card and ending with a
  local demo run.

- `architecture.md`  
  Current software/hardware/system architecture in plain language.

- `mapping.md`  
  How the repo folders map to the proposal and what was preserved or added.

- `checkpoints.md`  
  Milestone log. This is the file to update after each serious run.

- `manual.tex`  
  Short operational manual for the training, validation, CLI, and evidence
  workflow.

- `checklist.tex`  
  Submission/review checklist.

- `code_notes.tex`  
  Notes on important source files and responsibilities.

The progress report source lives in `paper_output/progress_report.tex` because
its PDF is also generated there.

## How I Keep This Folder

These notes should read like project notes, not generated filler. When adding a
new document:

- write what was actually run or checked;
- mention pending work honestly;
- keep commands copy-pasteable;
- avoid long abstract claims unless a result file backs them up;
- put plots/tables/logs in `results/`, not here.

