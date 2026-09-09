# CLI Entrypoint Contract

These are planned interfaces. The new script and campaign YAML will be created
during implementation. Examples do not authorize or launch training.

## Campaign preflight/materialization

```bash
python scripts/analyze_tinystories_optimizer_ownership.py preflight \
  --campaign configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml \
  --prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1 \
  --tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1 \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign \
  --run-output-root /scratch/ivo.navarrete/tmp/optimizer-ownership-runs
```

`--campaign`, `--prepared-corpus-dir`, `--tokenizer-dir`, `--output-dir` and
`--run-output-root` are required. The script audits immutable data and roles,
expands exactly nine arms, resolves normal configs without creating run output
directories, CPU-constructs models for counts/topology, computes expected complete
action/data digests and verifies exact controls. It writes:

```text
<output-dir>/
├── campaign_manifest.json
├── preflight.json
└── configs/
    ├── ST-g250.yaml
    ├── ST-g500.yaml
    ├── ST-g750.yaml
    ├── ST-g1000.yaml
    ├── S1.yaml
    ├── S2.yaml
    ├── C1.yaml
    ├── C2.yaml
    └── C3.yaml
```

The ordinary executable configs carry campaign identity and expected contract;
resolved controls are also retained in the manifest. `run-output-root/<arm-id>`
is the explicit output location recorded per run; run IDs include the campaign
ID, arm and seed. Fail with nonzero exit and named discrepancy on bad inputs,
occupied run identities or invalid controls. Do not publish a success manifest
for partially failed preflight; staged diagnostic failure output may be retained.
Preflight does no forward/backward, training or holdout model evaluation.

## Existing trainer entrypoint

Single-run config inspection remains:

```bash
python train.py --config /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign/configs/C3.yaml --preflight
```

Add resolved ownership/clipping/campaign identity to its JSON. This config-only
check does not replace campaign audit/model/trace validation.

After implementation verification and a separate campaign execution request,
each arm uses the existing trainer, with its manifest-recorded output directory:

```bash
python train.py \
  --config /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign/configs/C3.yaml \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-runs/C3
```

The same immutable config/output identity invokes existing continuation behavior
when a durable compatible checkpoint exists. No cross-arm source checkpoint,
changed horizon or historical run extension is accepted. There is no new command
that implicitly launches all arms as part of preflight or plotting.

If the run already has its durable terminal checkpoint but lacks the terminal
validation sidecar, the same trainer invocation resumes completion only, with
zero further optimizer updates and the same terminal checkpoint hash.

## Freeze terminal identities

```bash
python scripts/analyze_tinystories_optimizer_ownership.py freeze \
  --campaign-manifest /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign/campaign_manifest.json \
  --run-root /scratch/ivo.navarrete/tmp/optimizer-ownership-runs \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-frozen
```

`--run-root` resolves the nine manifest-recorded arm directories; alternatively
accept repeated `--run-dir` paths, mutually exclusive with `--run-root`. Validate
all expected identities rather than inferring arm identity from a directory name.
Write `frozen_manifest.json` only after terminal checkpoint, summary, sidecar,
complete controls, traces and endpoint validation. Never select best/trailing
metrics or automatically evaluate a model.

## Report

```bash
python scripts/analyze_tinystories_optimizer_ownership.py report \
  --manifest /scratch/ivo.navarrete/tmp/optimizer-ownership-frozen/frozen_manifest.json \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-report
```

Validate frozen sources again, write `comparison_report.json`, the prescribed
CSV/JSON endpoint exports, four combined image files and individual plots.
Default requires complete nine-run/24-point evidence and returns nonzero on any
mismatch. Reporting only consumes saved ordinary-validation artifacts.

Both freeze and report may accept explicit `--allow-partial` for a separately
requested diagnostic subset. Partial manifests/outputs list missing endpoints
and carry visible partial labels; report must also receive the flag to consume
a partial manifest. Malformed or mismatched supplied inputs still fail.

No CLI in this workflow evaluates sealed holdout or changes historical analyzer
semantics. Any future holdout comparison has its own explicit request and freeze
and evaluation requirements from the specification.
