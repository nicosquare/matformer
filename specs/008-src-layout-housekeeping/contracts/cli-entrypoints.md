# Contract: CLI Entrypoints

## Training Entry Point

### Command

```bash
python train.py
```

### Supported behavior

- With `--config`, the command uses the config-driven training flow.
- Without `--config`, the legacy direct-training path remains available.
- Existing flags and override semantics remain stable.
- The command continues to write the same classes of run artifacts as before.

## Figure Generation Entry Point

### Command

```bash
python scripts/make_figures.py
```

### Supported behavior

- Accepts the existing `--input`, `--output`, `--no-refresh-counts`, and `--dpi` flags.
- Reads the existing CSV artifacts and writes the same classes of figures and summary documents as before.
- Remains a thin wrapper around importable reporting helpers in `src/`.
- Continues to emit the per-panel companion PNGs for size plots alongside the combined figure outputs.

## Import Compatibility Contract

- The current package names stay importable as `models`, `training`, `evaluation`, and `utils`.
- Reporting helper code lives under `src/evaluation/reporting.py` and `src/evaluation/reporting_styles.py`.
- Root-level wrappers are not the canonical home for implementation logic.
- Tests and internal modules should import package code, not rely on filesystem-specific path hacks.
