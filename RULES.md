# Memory Compiler Workspace Rules

1. **Config Driven**: All experiments and runs are driven by YAML files in `configs/`. Every run must be seeded and strictly write its output to `runs/<exp_id>/{config.yaml,metrics.json,preds.jsonl,git_hash.txt,log.txt}`.
2. **Tests First**: `pytest` must pass before any training or baseline script runs.
3. **Data Split Discipline**: Never read or evaluate on the test split until `configs/FREEZE_TEST` exists. Train on train split, carve 10% validation split by video id (seeded, never by query).
4. **Honest Metrics**: Never invent numbers. All performance tables are auto-generated from `scripts/make_tables.py` reading actual run artifacts in `runs/`.
5. **VRAM Safety (4 GB Cap)**: Peak VRAM must stay under 3.5 GB. If out-of-memory occurs, reduce micro batch size and enable gradient checkpointing; never change the method silently.
6. **Artifact Traceability**: Each task completion produces an artifact containing the diff, test verification output, and recorded metrics.
