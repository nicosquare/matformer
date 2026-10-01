# Phase 1 compatibility verification

Date: 2026-10-01. Scope: T001–T002 only. No runtime source/configuration changes, external output reservation, historical artifact writes or GPU submissions.

## Environment and source identity

Interpreter: `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`.
Python 3.12.13; PyTorch 2.11.0+cu128; Transformers 5.8.0; NumPy 2.4.3; PyYAML 6.0.3; Matplotlib 3.10.9; pytest 9.0.3. No packages installed.

Repository HEAD: `e3d9a04e3a9ed94e671259e7259bd45f914c2bfd`. Initial working tree was clean. The hash manifest below binds the runtime, recipes, fixtures and tests inspected/tested in Phase 1; documentation is excluded to avoid self-referential hashing.

Git and Python setup verified. Existing `.gitignore` covers Python bytecode/environments/build artifacts, logs, secrets and universal editor/OS patterns. No Docker, ESLint, Prettier, Terraform, npm publishing or Helm setup detected; no ignore changes needed.

## Commands and outcomes

Historical baseline:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_config.py tests/test_optimizer_ownership.py tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_resume.py tests/test_optimizer_ownership_reporting.py tests/test_optimizer_ownership_corrections.py tests/test_optimizer_ownership_correction_queue.py tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py tests/test_c4_selected_block.py tests/test_c4_separate_corrections.py tests/test_c4_separate_reporting_queue.py tests/test_s1_warmup_campaign.py tests/test_s1_warmup_queue.py tests/test_s1_warmup_reporting.py tests/test_metrics_compact_accounting.py tests/test_metrics_history_performance.py tests/test_reproducibility.py -q -rs --tb=short
```

Focused compatibility:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_campaign.py -q -rs --tb=short
```

- Historical baseline: **1073 passed, 38 skipped, 2 warnings in 361.59s; exit 0**. Retained log: [evidence/phase1-baseline.txt](evidence/phase1-baseline.txt); SHA-256 `9e4f017e83b218b8b2231795f383f56a56c2a928220203d3ea58a9c6c3914388`.
- Focused compatibility: **36 passed, 0 skipped, 2 warnings in 9.48s; exit 0**. Retained log: [evidence/phase1-focused.txt](evidence/phase1-focused.txt); SHA-256 `0e2c81ed23bbe807a646b72dbe804f87b0c0776ce1ff126d5038e52549929885`.

All tests passed. Skips: 34 CUDA/BF16 optimizer ownership cases and four C4 CUDA gates; CUDA was unavailable. Both suites emitted the existing SWIG type deprecation warnings. No GPU readiness is inferred.

## Tested compatibility boundary

- Schemas 1–3 use the existing `optimizer_ownership_legacy_signatures.json` expected signatures unchanged. Schema 4 uses existing `s1_warmup_schema4_signatures.json`, including normalized resolved configuration hashes. Existing helpers check contract field sets, materialized resolution and YAML/JSON serialization without mutating contracts.
- Schema 5 freezes both warmup arms' full normalized resolved configuration and contract hashes in `test_linear_calr_campaign.py`. This new snapshot was captured from the unchanged baseline. Only provenance is pinned to `linear-calr-phase1-schema5-fixture`; exactly four fixture paths are normalized. This is separate from the historical expected signature fixtures, which were not regenerated.
- S1/shared, S2/per-granularity and C3/block-owner cosine clocks are probed at 0/63/64/65/87132/174264/348527/348528. Rates follow independent cosine arithmetic, with exact restored optimizer/clock state. Committed model/optimizer/clock bundle continuation is compared exactly after an epoch boundary.
- Separate C4 GMC/LMC-only corrections on linear/geometric grids reuse existing real-gradient/populated-moment continuation and incompatible-metadata rejection checks at warmup completion. The broad baseline additionally exercises all existing C4 schedule positions and pre-mutation rejection cases.

An initial focused test draft incorrectly supplied `resolved_total_steps` rather than the historical scheduler's `max_steps`; 15 assertions failed. The test setup was corrected without changing runtime code. The corrected suite results below supersede that draft.

## Limits

Corpus/tokenizer audits use existing fixtures; saved historical terminal checkpoints were not audited. Clock probes seed boundary states and do not represent full-horizon training. CUDA skips are separate from passes and establish no GPU readiness. Schema-6 contracts, polynomial/CaLR rates, real GPU diagnostics, production and comparison acceptance remain later-phase work.

## Source/configuration SHA-256 bindings

| Path | SHA-256 |
|---|---|
| `configs/controlled_exps/tinystories_controlled_convergence.yaml` | `b52aa9a057e388c41f337c2320b589c5a3b37ca5cac27bef2e54ab87ad2f794e` |
| `configs/controlled_exps/tinystories_instruct_c4_selected_block_geometric.yaml` | `24f617915e8e57193d6f0c8e3c3690bdd4be25c55d9c28057d5c548377536808` |
| `configs/controlled_exps/tinystories_instruct_c4_selected_block_linear.yaml` | `b5eb456da3acbded6d19d8d7d402a002895a505a5b2c7679a50d435aab4a9ed3` |
| `configs/controlled_exps/tinystories_instruct_c4_separate_geometric-GMC.yaml` | `0b77f54599f1ae1fc25860797ccb7981572b8debaf37f02343b247ff1e3fc81f` |
| `configs/controlled_exps/tinystories_instruct_c4_separate_geometric-LMC-only.yaml` | `7f55c1f69b3a61cc36d3cafb1d6890b0e4ebe292f706298eb01dc06d9546b742` |
| `configs/controlled_exps/tinystories_instruct_c4_separate_linear-GMC.yaml` | `bdbe69f801d1e2e17f0573f262de1c6da57587d116b6d3824938053743863b4f` |
| `configs/controlled_exps/tinystories_instruct_c4_separate_linear-LMC-only.yaml` | `835543af78767e8a049043ddb848fa7adfae80071325059ed4dce93316bad456` |
| `configs/controlled_exps/tinystories_instruct_inverse_membership.yaml` | `f4c964cee967531cfab51447ec10fb0b0e843a76902c944ebb7827b32c8822f6` |
| `configs/controlled_exps/tinystories_instruct_matformer_widths.yaml` | `e8a6774d002bfe983e308a4b319c077bbf4f2ba6e1908182fa5108afca0816d4` |
| `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml` | `4a3218bd745c74a0c8aed3610d1d2b09dec889bdafaf751781c2f4eb099c0bea` |
| `configs/controlled_exps/tinystories_instruct_optimizer_ownership_corrections.yaml` | `278671bf2a666623b38f91c1e41cebe483773f37e3dda2c298fde3365c2c659a` |
| `configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml` | `4ef6f264f53bf38f35626e14cabce0469c81c3f716a87c513ad66fc8158809f9` |
| `configs/controlled_exps/tinystories_instruct_plateau.yaml` | `67662c8e00ad377e625b68023257157e65227a263f8c46432a0c222536eaa098` |
| `configs/controlled_exps/tinystories_instruct_portfolio_catchup.yaml` | `ae2e907bbf8b88aaa6888abc9a5cc469db451a49575946b371dc2fd59d0c732e` |
| `configs/controlled_exps/tinystories_instruct_s1_warmup.yaml` | `133ba407d683e803c023a301680e78e9675bb4e269f436a77b38fbf82d858b6b` |
| `configs/controlled_exps/tinystories_instruct_sign_dynamics.yaml` | `f8992b19b6b170b736decc14e2d5039482dcc518b26f037da87d40f3486592c7` |
| `configs/controlled_exps/tinystories_instruct_wsd_calibration.yaml` | `6cf2c2dc94cd34fed2aaaf066c8b9a7a9347d9cd8b918a86c742605d2006f573` |
| `src/__init__.py` | `981a0840b0aeb0a81b4453143e2333e6026d57b945ad8e01b3740c70472fd36c` |
| `src/evaluation/__init__.py` | `d7000d1a9a855a5a0968dd8aab9f64d3650735a658691bbd2633b602229ee4c2` |
| `src/evaluation/consistency.py` | `c5f2becef2abfb651fbcbc07d0c654ec86d6e85128dc58365de071d787f599a2` |
| `src/evaluation/downstream.py` | `ff99ed8cb2a93f7aaa25df269bf4db6d36821aa5828a82f13e501815abddc526` |
| `src/evaluation/final_holdout.py` | `f9579689c69cc578f99ec9c60f20d8800205a3fb350a1d17d888e6c59e23eebd` |
| `src/evaluation/optimizer_ownership.py` | `fc936f7f6044b184f37bfa5856d0baafc04b0d0a2ae7ffda729e44f3fa1a234c` |
| `src/evaluation/reporting.py` | `e8ac52fa45603793a54682a5e228798591df91c1d31cc93013c632c456229881` |
| `src/evaluation/reporting_impl.py` | `dafebf4046e13dc4c62cb96fe68d33057ea564683e2281a3d3704fa800ed7b7e` |
| `src/evaluation/reporting_io.py` | `e776332fa5b519509c393ab8b0e52faec03d6e815cba0a83caaa1212ead3ea99` |
| `src/evaluation/reporting_styles.py` | `704e52af0cad6389466c41678cf924c29a0932cad4f766efcbda25b8e5e02182` |
| `src/evaluation/speculative.py` | `3f9f67cdacceb52e0bd190d5101837d05039d4d5dd8a500a9ac4911da02529f8` |
| `src/evaluation/validation.py` | `452ae96a62699b99c9818c112a2c7573bb69cac30ea93abc1ba5c9ba695b301c` |
| `src/models/__init__.py` | `178ac182a914f43b326471c1543334b2445625ba93537ba946bc13f89892e0fc` |
| `src/models/adaptive_sampler.py` | `1b300fb3b4fd9b5e98fd0bce58b634bf5fcfcd87107e292234a9efb425486429` |
| `src/models/correction.py` | `10d6612d301ad55cb9fad2c9573336d3786a1e9d6c28fe1a7254afa997713d13` |
| `src/models/ffn.py` | `e568ae79bb8cb14cd325d12bf099695265b8240a4370afbd9d2f81dd7cfa00ad` |
| `src/models/granularity.py` | `88b31e0c6f60e03f49ac2b52777bd8be325e5c0ceb8727eb6743a9d63080a439` |
| `src/models/wiring.py` | `b6d4381fe313f1343987cbcafc509b8f54d43ab12922bcb5d54ca06a750858aa` |
| `src/training/__init__.py` | `c9992966495aee9958246dfd25f1d6692b61bb26a4804c27c588e108cf84b6d1` |
| `src/training/baselines.py` | `6f3e29d7e4d0966319697b3a6b5a478e0c03b1efcff40fd783a6bc0cfa2296f7` |
| `src/training/checkpointing.py` | `c4f6cf5f22a30895182bea57bade2d94b280115699a57fb0b7b6b11c60cca541` |
| `src/training/data.py` | `128e53df6a8c0db896e1b75bc90b14769e323d853cb3ed6253864f168f813726` |
| `src/training/distributed.py` | `7dcd22de96124f5e423bdbea8cc3a8ed210d0de8c0fd90d8fa4aa80248e98c6b` |
| `src/training/fineweb_tokenizer.py` | `7dac28fbd921b65ed91853b56442fbfe4d35ed46ac70d6cf138ad422b2dd07c9` |
| `src/training/gradient_interference.py` | `44613fb78ee4926b47f67df9c9a1f8abf22e01ea8f26bcf65258d1e607c31ea2` |
| `src/training/gradient_probe.py` | `15371e485fb3c18c7ffb9495875ecb60a13345c34230016ba22c42f2365ff81c` |
| `src/training/modeling.py` | `6ba1672862b90ad2924b064d215d2f925550db153dc9f834da979228b0fa5048` |
| `src/training/monitoring.py` | `e32ac302bd3364e2a769b6824518a240b607cc9870518f17f3e88b83307023df` |
| `src/training/optimizer_state.py` | `411f7460db49caca4a3fd92a61f4d05b2f99ccfeb1ee4cf4e7e8f4f502e5b23a` |
| `src/training/packed_corpus.py` | `164c09552b422a88fe47178b1f8327136084ceabe00e72e86655c0810a47c2a4` |
| `src/training/panelgrad.py` | `f23dd3ec031fbbd6219b89827ac93f7ae7899ed82f05d914f8d4113e2bc756ab` |
| `src/training/portfolio_catchup.py` | `b5f24aff8163e66cce22aa768ae435cb0cbaaedfbc7943b4e2704f21ba750b8d` |
| `src/training/probabilistic_controller.py` | `ef266191fdec1f69170b7c6151d07ed462e31ac057f8d8e737c11254a60bc4e3` |
| `src/training/run.py` | `ea8deee151ff9d91a0d0b1492744cf397a9c1687190456724128dfb5b45e5f7f` |
| `src/training/schedules.py` | `dde7be2a550fb1809e9532b732514acc2583b04eb160e13a7a231ea64653a5c7` |
| `src/training/sign_dynamics.py` | `67c167ab935a4762cec9c65cedf3bd35a831f554cbbe3048223fe7f89adfa860` |
| `src/training/steps.py` | `2097bb74e243008b653e7cf564bbb0cc396a71694521604f939efcd8de037df1` |
| `src/training/tinystories.py` | `7413ca22e719ee59b20bdb935b084050daad093ef9036574eb2eb731d98c57a9` |
| `src/training/tinystories_instruct.py` | `b7418a8d55c2662c79f4a31c3e02c4c002f24c167a9be50542e2c67679e16be4` |
| `src/training/tinystories_preparation.py` | `0e3960e3fcb8bb22772933a4c4f41d9bf1eff04a3868ee36e9c30528b232024a` |
| `src/training/warmup.py` | `7c531de628c7968e61342db115d7851538a1278db1e22b91b4149cd2a1ce17ef` |
| `src/utils/__init__.py` | `29d28af6cfe9fff76c572dc6362cb5c74e13affc464c057ea8efc5ea54703634` |
| `src/utils/artifact_io.py` | `477a8669c48e1be19166a708ed3ab58fe6dc21f550f5d6144ee6ef3bc9b896b7` |
| `src/utils/config.py` | `ba81bb871a5bb95ca5a384b6a35d896f8e03a8b6f36329aa3970f634f36c5e0a` |
| `src/utils/heartbeats.py` | `6d10b71b812942898ad3f10d8fa0403ba2ea12d2e11b0fecd9518f6b084c5006` |
| `src/utils/metrics.py` | `431924d8ec7e5c10ad189664aedc0ee4bbb4f6057c58448ae799d55263a22cb7` |
| `src/utils/model_size.py` | `6642b1875e3e9591f1addc02bcc3535f0b96edfa7b1df08327c3f8694c8c593e` |
| `src/utils/monitoring.py` | `d973413540e586ce1f6c2de504d56eebfa7dca366ded2f44395c039ac2911c3b` |
| `src/utils/reproducibility.py` | `88dc41965d46220624ebf443772804af2ff4c0f1431ea2993be5a7e8e84442f7` |
| `tests/fixtures/optimizer_ownership_legacy_signatures.json` | `8a3d67fac6f7bc266866bc24b597de71dadb7b5dc1d0af0f54c914ce74201252` |
| `tests/fixtures/s1_warmup_schema4_signatures.json` | `657031baaf15b720605259799b6ba9b89fcb7e2d721e4fd6b7df3030b1100b19` |
| `tests/test_c4_selected_block.py` | `cb6fce4b0e3bd3bc7e23136de1884853a846736145d7735e88818d3de2c5d472` |
| `tests/test_c4_separate_corrections.py` | `1ce01d17e3bb4956dcaedec635dd8b85d4ab48db9cbce50a32184fd63755b580` |
| `tests/test_c4_separate_reporting_queue.py` | `7cfe7fb4471d91712c03849173b783a7df75ae78e8096459d451a22423e74260` |
| `tests/test_config.py` | `236a01d709a381931e11ee94bf6cd29bfc1a2d47b6cbd135495bb46664cc3096` |
| `tests/test_linear_calr_campaign.py` | `ccf5380c203484185d90ca9be8083e174a77dd68fcb1fbe1a7a91cb3842f0a71` |
| `tests/test_metrics_compact_accounting.py` | `63f1c5cc33e3fc5c5e7de9a1c0f4a52ae6342cd1bb8dfa4feb984c80aae0758a` |
| `tests/test_metrics_history_performance.py` | `1fdcf9b047b8af170072f2b94c921062149bf5ec4d1abe1abdb4fdab83212eb5` |
| `tests/test_optimizer_ownership.py` | `0ea663dfa706ecc08ae4d99351e72b1a5b03eb9958760abe17e888c261aa16b8` |
| `tests/test_optimizer_ownership_campaign.py` | `80e6fc0bdfebd01df25ed77897d771173d2367d23a434085dda15ce3a2fb40d5` |
| `tests/test_optimizer_ownership_correction_queue.py` | `d7f0a6a702ef533689d3c749f6e8010c323cb45b9087b5b6303facda1d46c9da` |
| `tests/test_optimizer_ownership_corrections.py` | `371bc00c4a67588c4210107ccef0c1d19a23b1b604541de65c539bc381760604` |
| `tests/test_optimizer_ownership_reporting.py` | `5341e2c991f85d38082cdd4abafd82738b72be74ef31aa93fd66826d1d3be5de` |
| `tests/test_optimizer_ownership_resume.py` | `fcbb64d2e6775ff6ebba2964fd23a0f378cdbffb0bee294a8d05921903732fc7` |
| `tests/test_per_granularity_optimizer.py` | `157287d440023ce3ee71ef2cafbbdf9e1770d4f96c0b6fcbcf1732c73b5382de` |
| `tests/test_per_granularity_optimizer_resume.py` | `6be496a7762ed2e20fff00270ee1aa447c563481feca4b86fda32a213271ed66` |
| `tests/test_reproducibility.py` | `bbbaa0e667cbc03e047402693889d4dc76b2c3320dc40766b4e27fd4bc9f2382` |
| `tests/test_s1_warmup_campaign.py` | `9e8d960d3ac620071323579870526c5140328634f7d90650baf700fb648827c9` |
| `tests/test_s1_warmup_queue.py` | `4bac493b0976e72937b499d4c361eb606059e6a162950ef5f221ca39ef2d4062` |
| `tests/test_s1_warmup_reporting.py` | `7d375dfbedf1b8a06aa9942f8491625cb7375b95b4a43e6d7eb996a4dcdccf66` |

## Phase 2 foundation verification

Date: 2026-10-02. Completed T003–T005. Added deeply immutable version-1 contract parsing with detached JSON/YAML serialization, semantic slicing/shared-or-per-granularity eligibility, new-only scientific and optimizer schedule/hash bindings, pure warmup-polynomial rates, complexity-log exponents, and explicitly analytic evidence fields. Named seed derivation is unchanged. Existing ignore patterns remain sufficient; no additional tool-specific setup was detected.

Command:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_foundation.py tests/test_linear_calr_campaign.py tests/test_config.py tests/test_reproducibility.py -q -rs --tb=short
```

Result: **362 passed, 0 skipped, 2 existing SWIG warnings in 15.21s; exit 0**. `git diff --check` passed. Retained log: [evidence/phase2-foundation.txt](evidence/phase2-foundation.txt).

Foundation tests cover immutable/detached contracts, malformed fields/counts/policies/positions, all-width warmup/decay/terminal anchors, finite nonnegative rates at every position of the full horizon, semantic eligibility for both scopes, changed identity hashes and name-independent initialization/action/data seeds. Historical tests preserve schema 1–5 signatures and S1/S2/C4 schedule/restore behavior.

This phase supplies primitives only: schema-6 campaign expansion, actual-model count validation, runtime LR transactions, checkpoint integration and operational readiness remain later tasks. No training, GPU jobs, external root reservation or historical writes were performed.

Tested SHA-256 bindings:

| File | SHA-256 |
| --- | --- |
| `src/utils/config.py` | `27f70d9e201ad1a176ac73b6ae485d789653fae1dbecec71efbc789bc6544d24` |
| `src/utils/reproducibility.py` | `d947470c25f311cb1e19afb7e9339f4b36641da81e8a8efff087e60394e4d974` |
| `src/training/schedules.py` | `b8f56505d3e1db685522df1cda0e03bbade062c468b557d26080d3be5619f1b8` |
| `tests/test_linear_calr_foundation.py` | `c7bc077384b12e4e05088a2b78e0acfeb4616467aa0e6f576d6d195c3ffb6e76` |
| `tests/test_linear_calr_campaign.py` | `ccf5380c203484185d90ca9be8083e174a77dd68fcb1fbe1a7a91cb3842f0a71` |
| `tests/fixtures/optimizer_ownership_legacy_signatures.json` | `8a3d67fac6f7bc266866bc24b597de71dadb7b5dc1d0af0f54c914ce74201252` |
| `specs/017-tinystories-linear-calr/evidence/phase2-foundation.txt` | `515c6c5490c4479dfd929733c2a1e2237f9fe93fc85cd5c0718e849e5be730d1` |


## Phase 3 — matched comparison definitions (T006–T013)

Date: 2026-10-02. Phase 3 is complete. Four schema-6 arms resolve without training:
S1-linear-poly, S1-linear-CaLR, S2-linear-poly, S2-linear-CaLR. Every run is
348528 updates / 2855141376 packed tokens; campaign totals are 1394112 /
11420565504. The recipe pins seed 42, .008 peak, 64 warmup, linear prefixes,
original controls/data identities, and all eight explicit saved reference paths.

Actual CPU models validate trainable active counts (including embeddings/head,
tied parameter identities once) 377408/426560/475712/524864 independently of
reporting counts 115264/164416/213568/262720. Preflight hashes fresh tensors and
requires equality across all four arms and original counterparts. Tests walk the
entire 348528-update runtime sampler horizon with bounded memory and compare
all action digests, batches and fixed epoch sets across all six definitions.
Closed resolved-field audits separately admit counterpart, schedule-policy and
ownership changes, including contract hash consequences. Disabled diagnostic
fields absent from old originals are admitted only at exact validated defaults.

The existing `preflight` CLI accepts schema 6 with required `--reference-root`
(the base directory containing the eight relative saved run mappings). Other
required options remain `--campaign`, `--prepared-corpus-dir`, `--tokenizer-dir`,
`--output-dir`, `--run-output-root`. It audits corpus/tokenizer/manifest controls,
counts, fresh weights, full traces and full-horizon analytic rates before atomic
publication/reservation. Failures return nonzero. Analytic CSVs include every
position 0 through T and every width, with separately named nominal/effective
rates. No field claims those rates were observed during training.

Commands (repository root, existing interpreter, no new packages):

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_campaign.py tests/test_linear_calr_schedule.py tests/test_linear_calr_foundation.py tests/test_config.py tests/test_optimizer_ownership_campaign.py tests/test_s1_warmup_campaign.py tests/test_reproducibility.py -q -rs --tb=short
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_campaign.py tests/test_linear_calr_schedule.py tests/test_linear_calr_foundation.py -q -rs --tb=short
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_campaign.py -k 'preflight_saved or counterpart_predating or rejects_changed_declared' -q -rs --tb=short
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python specs/017-tinystories-linear-calr/evidence/phase3-saved-controls.py
```

- Affected regressions: **539 passed, 0 skipped**, 117.16s, exit 0.
- Final focused formula/campaign/foundation suite: **93 passed, 0 skipped**,
  59.15s, exit 0. Includes frozen schema 1–5 signatures, S1/S2/C3 clocks/resume
  and separate C4 correction probes.
- Subsequent saved-config closure checks: **6 passed, 49 deselected, 0 skipped**,
  10.36s, exit 0. These cover the final read-only runtime-metadata checks added
  after the broad suite; no historical resolver branch changed afterward.
- Read-only saved discovery inspected exactly eight configs. All four new arms
  passed closed controls against actual original S1/S2 prepared definitions.
  The retained helper uses mocked corpus metadata and normal CPU models; it
  verifies saved controls, not the live corpus or terminal checkpoints.

Only existing SWIG deprecation warnings were emitted. Initial test-first
failures identified the missing new recipe/API and new campaign output identity;
those were resolved. Real saved config discovery exposed the existing runtime
world-size source and role-manifest additions. These are checked explicitly,
including exact role hashes; unknown model/training/evaluation controls fail.

Limits: no GPU work, external root reservation, training or historical writes.
No terminal checkpoint/reference acceptance is claimed. Fixture preflight uses
mocked corpus/trace I/O and short analytic CSVs for publication checks; separate
full-horizon tests verify every rate and every runtime sampler step. Actual
corpus reaudit and readiness remain later tasks. Runtime rate application and
continuation for schema 6 remain Phase 4.

Retained evidence and SHA-256:

| Artifact | SHA-256 |
|---|---|
| [evidence/phase3-regressions.txt](evidence/phase3-regressions.txt) | `f96108185cd4c332fa75b1f7f0ebb6846b37eac2f075799d39f3b56e51c2ab86` |
| [evidence/phase3-focused.txt](evidence/phase3-focused.txt) | `879b476b9621126073dcd885b3bb4d26a8e71a0c934836148e02f0fca7326202` |
| [evidence/phase3-reference.txt](evidence/phase3-reference.txt) | `5b82bf4ba3b5ecda5a67bf589890f316bcfbbd3ea0511472527009ca9b124b21` |
| [evidence/phase3-saved-controls.txt](evidence/phase3-saved-controls.txt) | `0d66866e15de3736f5d934b8b0505f886eaccf99d3f0315f2e54cb383c81cfb1` |
| [evidence/phase3-saved-controls.py](evidence/phase3-saved-controls.py) | `c3a80f57aaafde9760dc4308c94447ef62ac8e66fe01d176f42922540dfe63de` |

Final source/config/test bindings:

| Path | SHA-256 |
|---|---|
| `configs/controlled_exps/tinystories_instruct_linear_calr.yaml` | `9425b2736f8e12ec0c2c0028f96f3a9d39d79c044248e5f319b58e6913374b72` |
| `src/evaluation/optimizer_ownership.py` | `494c134c3c2d387b4836835afc304005591773ef5335908e4c9b6038b9d9c35b` |
| `src/utils/config.py` | `c862ee181897778521773421faaf2ba1436dd73ae99864c18719583c5352b246` |
| `src/utils/reproducibility.py` | `d947470c25f311cb1e19afb7e9339f4b36641da81e8a8efff087e60394e4d974` |
| `src/training/schedules.py` | `b8f56505d3e1db685522df1cda0e03bbade062c468b557d26080d3be5619f1b8` |
| `src/utils/model_size.py` | `6642b1875e3e9591f1addc02bcc3535f0b96edfa7b1df08327c3f8694c8c593e` |
| `scripts/analyze_tinystories_optimizer_ownership.py` | `338658ee1f60acc4e04b13435b1f5c281c1bb3dab6c429d8da6f5246c7b1c9fe` |
| `tests/test_linear_calr_campaign.py` | `ce9c8ad529aff3b3b99d2c79bf048f49631af8c640fd5e284af5a0620bbf56cd` |
| `tests/test_linear_calr_schedule.py` | `65d0c915bb6a6a5880e9ac236aa3b0272858876561ad0f3fc99d6f4b87aa5aaf` |
| `tests/test_linear_calr_foundation.py` | `c7bc077384b12e4e05088a2b78e0acfeb4616467aa0e6f576d6d195c3ffb6e76` |
| `tests/fixtures/optimizer_ownership_legacy_signatures.json` | `8a3d67fac6f7bc266866bc24b597de71dadb7b5dc1d0af0f54c914ce74201252` |
| `tests/fixtures/s1_warmup_schema4_signatures.json` | `657031baaf15b720605259799b6ba9b89fcb7e2d721e4fd6b7df3030b1100b19` |

## Phase 4 implementation and CPU verification (2026-10-02)

Scope: **T014–T028 complete**. T029–T031 remain pending their separate external-preparation, diagnostic and production authorizations. This invocation created no external campaign root, authorization record, GPU job or production result. No saved reference was modified. Existing ignore patterns cover this Python project; no additional tool-specific setup was detected.

The implementation uses the existing trainer, GlobalSchedulerClock, optimizer collection layouts, fatal mutation boundary, checkpoint bundle staging, repeat sampler, terminal recovery, source snapshot, locks, submission intents, resource ledger and live Slurm admission helpers. New behavior is conditional on the immutable polynomial schedule contract. Historical signature fixtures were not regenerated.

| Tasks | Completed verification/behavior |
|---|---|
| T014 | All four arms match manual AdamW updates with populated moments, decay, two parameter groups, all widths and shared physical parameter identities. p=0 preserves weights while advancing history. Unselected S2 states are unchanged; full-shaped sliced-tail gradients are zero and tails retain inherited decay behavior. Initial/group options are preserved; terminal updates are rejected. |
| T015 | Exact own-arm continuation before/at/after warmup, representative decay, each 87132-update epoch boundary and terminal T. Full-horizon probes seed synthetic coherent histories and a real repeating sampler over a virtual dataset; only the subsequent diagnostic updates are executed. Weights, histories, counters, RNG, sampler/cursor, nominal clock, applied record and chain watermark match exactly, stricter than rtol=1e-6/atol=1e-7. |
| T016 | Cross-arm/grid/scope/count/gamma/horizon/record/watermark/clock/RNG corruption is rejected before live mutation. Optimizer, restoration, scheduler and accounting failures poison state and preserve the last durable checkpoint. Missing terminal output recovers with zero optimizer steps. |
| T017 | Fixture-only lifecycle tests cover source/config/reference/log changes, occupied roots, absent/stale durable authorization, failed/stale/missing gates, CUDA fallback rejection, duplicate/uncertain submission, stricter user-wide limits, worker/Slurm mismatch and completion-only continuation. Four arms are admitted in two conservative waves. |
| T018–T019 | One exact gamma-one nominal polynomial clock for both scopes; temporary selected effective LR on every stepped group. Actual rates are captured inside the unsafe interval; finally restoration precedes clock advancement and synchronization. Success publishes only reconciled evidence. |
| T020 | Bounded committed JSONL records carry applied width/owner/count/gamma/position/all-group rates alongside existing action/data/exposure/resource evidence. Checkpoints retain only last record and chain watermark. Ordinary validation, update work and failed-attempt process costs are separately measured, overlap total process time and are never summed into it. |
| T021–T022 | Entire own-arm bundle validated before installation; nominal reconstruction and step−1 applied evidence are separate. Restore/admission streams the durable trace prefix and validates its chain/watermark. Terminal validation rechecks the own full-budget bundle and recovers missing outputs without training. |
| T023 | Atomic fresh snapshot, frozen source/config/input bindings, full corpus/control/model/action/data/schedule audits and executed CPU suites. Per-attempt failed records/log hashes are retained. CPU never submits jobs; changed input flags cannot replace an existing snapshot. |
| T024 | CUDA-required diagnostic worker implemented, **not executed**: every real d64/l4/h4, batch-64/context-128 arm runs 64 inherited real-corpus updates plus own continuation and proves actual BF16, all-width/nonzero-history coverage and measured resource/storage evidence. Separate full-horizon seeded CUDA probes use the same real shape and validate schedule/epoch/terminal continuation; invalid restore and failure tests also run under CUDA. Skips cannot pass the GPU gate. CPU exact tensor assertions are retained for the later same-device probes; no device-specific outcome is claimed. |
| T025–T027 | Fresh four-run preparation and report dispatch; independent readiness/submission/execution/terminal/comparison states. Durable subsequent authorization is separate from gates. Shared helpers enforce partition/QoS, exclusions, one GPU/no requeue, two-running/four-submitted or stricter limits, reconcile uncertain jobs and validate own continuations. Completion requires both worker and Slurm evidence and exact 348528/2855141376 per-arm and 1394112/11420565504 campaign budgets. Recorded completions are revalidated on restart. |
| T028 | Focused and historical CPU verification below; source/config bindings retained. |

The launcher dispatches `report-linear-calr`, whose implementation remains T039 in Phase 5; a missing/failed analyzer records incomplete comparison status. CPU evidence deliberately keeps `reporting_fixture_status=pending` until Phase 5 supplies executed reporting acceptance. Production cannot bypass that dependency.

### Exact commands and results

Broad compatibility run:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_foundation.py tests/test_linear_calr_campaign.py tests/test_linear_calr_schedule.py tests/test_linear_calr_resume.py tests/test_linear_calr_queue.py tests/test_config.py tests/test_optimizer_ownership.py tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_resume.py tests/test_optimizer_ownership_reporting.py tests/test_optimizer_ownership_corrections.py tests/test_c4_separate_corrections.py tests/test_c4_separate_reporting_queue.py tests/test_metrics_compact_accounting.py tests/test_reproducibility.py tests/test_s1_warmup_queue.py -q -rs --tb=short
```

**1171 passed, 70 skipped, 2 warnings in 240.88s; exit 0**. Log: [evidence/phase4-regressions.txt](evidence/phase4-regressions.txt). Skips are 32 separately authorized new CUDA probes, 34 inherited CUDA/BF16 ownership cases and four C4 CUDA gates. Warnings are existing SWIG deprecations.

Final focused run, after the schema-6 saved-manifest reader and restart reconciliation changes:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_foundation.py tests/test_linear_calr_campaign.py tests/test_linear_calr_schedule.py tests/test_linear_calr_resume.py tests/test_linear_calr_queue.py -q -rs --tb=short
```

**317 passed, 32 skipped, 2 warnings in 102.71s; exit 0**. Log: [evidence/phase4-focused.txt](evidence/phase4-focused.txt). The skipped CUDA probes establish no readiness.

Final affected-file checks:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_queue.py -q -rs --tb=short
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_schedule.py -k actual --tb=short -q
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python scripts/preflight_tinystories_linear_calr.py --help
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python scripts/run_tinystories_linear_calr.py --help
git diff --check
```

Queue: **82 passed**, 2 warnings, 7.70s; [evidence/phase4-queue.txt](evidence/phase4-queue.txt). Two-group actual updates: **4 passed, 15 deselected**, 2 warnings, 10.07s; [evidence/phase4-all-groups.txt](evidence/phase4-all-groups.txt). Both CLI help calls, Python syntax checks and whitespace checks passed. These commands operated only on repository/temporary fixtures and contacted no scheduler.

An initial broad run reported **39 failed, 1110 passed, 70 skipped**. Thirty-six newly added seeded fixtures lacked their live sampler attachment when saving; three inherited warmup CLI error cases exposed the Phase 3 misplaced `reference_root` argument. The fixtures and CLI dispatch were corrected; the passing broad/focused checks supersede that run. The failed log is retained as [evidence/phase4-initial-regressions.txt](evidence/phase4-initial-regressions.txt). A final actual saved-manifest fixture additionally identified and corrected schema-6 run-marker handling in the reader; all final focused checks include that correction.

### Final bindings and limitations

[evidence/phase4-bindings.json](evidence/phase4-bindings.json) binds all 236 source/config/test/entry-point files (documentation excluded). Source-set SHA-256: `d737d5e0bab55f71d97e016e389bb83c2d0c501f67905cfa162cb1ef50675fd6`. Schema-6 recipe SHA-256: `9425b2736f8e12ec0c2c0028f96f3a9d39d79c044248e5f319b58e6913374b72`. The broad run preceded the small saved-manifest/restart changes; the affected queue and complete focused suites passed after those changes. A final GPU-only output edit adds an explicit measured diagnostic throughput field; its worker has not been executed. The affected snapshot/diagnostic/GPU gate tests were rechecked separately (see phase4-gate-recheck.txt).

This is code/CPU correctness evidence, not a passing external snapshot gate, all-arm GPU readiness, real terminal training or comparison acceptance. Real corpus/reference reaudits, output reservation and actual CUDA/production execution remain T029–T031; Phase 5 reporting fixtures and terminal/reference provenance remain separate tasks. Synthetic seeded probes do not imply completed epochs or training.

Final GPU-output/gate recheck: `OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_linear_calr_queue.py -k 'gpu_gate or snapshot or diagnostic_submission' -q --tb=short`: **5 passed, 77 deselected**, 2 existing warnings, 6.15s; [evidence/phase4-gate-recheck.txt](evidence/phase4-gate-recheck.txt). Python syntax and final whitespace checks passed. This remains mocked gate verification, not GPU execution.


## Authorized T029–T031 execution

User instruction: `$speckit-implement authorized for T029–T031`. Separate diagnostic and production records retain this verbatim instruction and source/config bindings. Authorization remains distinct from passing readiness.

### T029 initial preparation

Created `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1` using the inherited corpus `/nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1`, tokenizer `/nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1`, and read-only reference root `/nfs-stor/ivo.navarrete/results/elasticnn`. Ran `snapshot`, frozen `cpu`, and frozen `prepare --cpu-evidence ROOT/diagnostics/cpu-gate.json`. The CPU command used `OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1` and the specified interpreter.

Full corpus/tokenizer/model/control/action/data audits passed and exactly four fresh identities were reserved. Eight references reached `config_inspected`, not terminal validation. CPU readiness: **949 passed, 66 skipped, 2 warnings in 153.06s**, exit 0; CUDA skips remain separate. Initial evidence: [phase4-external-preparation.json](evidence/phase4-external-preparation.json).

### T030 retained first failure and correction

Frozen `submit-gpu` admitted job **290522**, `calr-diagnostic-a1-29f1a00f`, with one GPU, declared partition/QoS/exclusions/no-requeue and verified controller limits (two running/four submitted). Scheduler accounting reports **FAILED, exit 1:0, 18 allocation seconds**. The diagnostic stopped before training: its output directory suffix differed from the canonical run ID. The probe now uses a separate diagnostic parent with the canonical run ID as its final path component. The actual campaign configuration test now enables output creation, exercising this invariant; **1 passed, 169 deselected**, 2 warnings, 6.60s.

The default accounting endpoint `localhost:6819` refused connection. Read-only reuse of the preceding campaign's verified `SLURM_CONF=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/launchers/slurm-client.conf` returned the scheduler proof. No historical or system configuration changed.

The entire initial root, immutable source, gates, authorization, reservation, logs and failure were retained at `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1-diagnostic-failed-290522`. No production state exists. A fresh corrected snapshot at the canonical root requires fresh CPU and GPU evidence; the initial gates are superseded. Failure evidence: [phase4-diagnostic-failed-290522.json](evidence/phase4-diagnostic-failed-290522.json), [worker traceback](evidence/phase4-diagnostic-failed-290522.txt).

T031 authorization is retained, but production additionally requires passing T030 and executed reporting-fixture acceptance T032–T041. Those Phase 5 tasks remain pending; the current CPU gate intentionally records `reporting_fixture_status: pending`.


The server restarted during the corrected snapshot's CPU coordinator. Its surviving subprocess finished **949 passed, 66 skipped, 2 warnings in 153.98s**, but no coordinator gate was published. The retained attempt is `diagnostics/cpu-8cc218d15119473e89b216a2d16e4938`; these test results are not promoted into readiness. After checking for live processes/jobs, a fresh frozen `cpu` invocation was started. The corrected immutable source hash is `0f8005347b1da937d561411e06bbe060fa77da47a92004ec9174cc5f61bda14f`; both authorization records match it.


### T029 corrected preparation and T030 complete

The fresh coordinator completed **949 passed, 66 skipped, 2 warnings in 149.21s**, exit 0, and published a passing CPU gate. Frozen `prepare` then reserved exactly the four new runs. Evidence: [corrected preparation](evidence/phase4-external-preparation-corrected.json), [CPU test log](evidence/phase4-corrected-cpu-tests.txt). T029 is complete for this corrected snapshot.

Exact operational commands after snapshot, using `ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1` and `PYTHON_BIN=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`:

```bash
export SLURM_CONF=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/launchers/slurm-client.conf
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$ROOT/source/scripts/preflight_tinystories_linear_calr.py" cpu --campaign-root "$ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$ROOT/source/scripts/run_tinystories_linear_calr.py" prepare --campaign-root "$ROOT" --cpu-evidence "$ROOT/diagnostics/cpu-gate.json"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$ROOT/source/scripts/preflight_tinystories_linear_calr.py" submit-gpu --campaign-root "$ROOT"
```

The final command was repeated after job completion for durable scheduler reconciliation; it returned passing evidence without submitting another job. Job **290529**, `calr-diagnostic-a1-4c37158d`, ran on **gpu-01 / NVIDIA A100-SXM4-40GB** and reports **COMPLETED, exit 0:0, 130 allocation seconds**. Partition/QoS, one GPU, no requeue, excluded nodes and live two-running/four-submitted ceilings were verified by the submitter and worker. The immutable snapshot, configurations, CPU gate, GPU gate, intent, worker and scheduler proof agree.

All four real-corpus probes used d64/l4/h4/vocab2048, batch64/context128 and actual CUDA BF16 forwards. Each performed 32 updates, saved its own checkpoint, resumed and reached64; the full horizon348528 was retained. Every arm observed64 BF16 forwards,524288 committed tokens, and width counts g250=9,g500=17,g750=21,g1000=17. No probe is a production terminal.

| Arm | Optimizer tensor bytes | CUDA peak allocated bytes | CUDA peak reserved bytes | Diagnostic tokens/s |
| --- | ---: | ---: | ---: | ---: |
| S1-linear-poly | 4199068 | 453779968 | 528482304 | 170467 |
| S1-linear-CaLR | 4199068 | 453779968 | 528482304 | 237615 |
| S2-linear-poly | 16796272 | 466390528 | 541065216 | 181805 |
| S2-linear-CaLR | 16796272 | 466390528 | 541065216 | 193100 |

Throughput divides64 committed updates by the measured trainer-process time including own resume/setup; it excludes allocation overhead and is not a production speed comparison. Deliberate diagnostic checkpoint stops are retained as interrupted attempts by the resource ledger; their costs are neither discarded nor added a second time to total process/allocation costs. Ordinary validation cost here is zero. The failed initial allocation's18 seconds remain separate.

The CUDA pytest command executed in the same allocated process:

```bash
"$PYTHON_BIN" -m pytest -q -rs --tb=short -p no:cacheprovider --junitxml=DIAGNOSTIC_OUTPUT/pytest.xml tests/test_linear_calr_resume.py tests/test_linear_calr_schedule.py -k 'not full_horizon and not all_widths_full_horizon'
```

**151 passed, 38 deselected, zero skips, 1 warning in98.79s**, exit0. The warning concerns an already-imported anyio module's assertion rewriting. Full-horizon analytic checks remain in CPU readiness. CUDA cases include all32 arm/boundary combinations at63/64/65,87132/174264/261396,348527/348528; these use explicitly synthetic state-seeded histories. Continuation weights/moments/clock assertions use exact `torch.equal`, stricter than inherited rtol1e-6/atol1e-7; no tolerance was relaxed for A100. Actual updates, invalid restores, poisoning, terminal recovery and all-group checks passed.

Current source SHA-256: `0f8005347b1da937d561411e06bbe060fa77da47a92004ec9174cc5f61bda14f`. Config-set SHA-256: `416efb09ad596e966aba904942daa8d279562417ec9119016498e9485cf8b6d0`. CPU gate content hash: `bcc8cbaf6a7c88badedf67867f5bbdd43baec5ccd816172294f6830d8d9124ad`. GPU gate content hash: `6fe4b3450bdf607ce770191cc0a41f01a23667cc384a9e6843a065e1eded5ff7`. Complete executed commands, logs/artifact hashes, measurements, environment and scheduler/worker records: [CUDA readiness](evidence/phase4-cuda-readiness.json), [CUDA test log](evidence/phase4-cuda-tests.txt).

### T031 remains pending: reporting prerequisite

With the durable production authorization and both passing gates, `verify_plan(ROOT, gpu=True)` returns **Production requires phase-5 reporting fixture acceptance**. T032–T041 have not been implemented/executed; `reporting_fixture_status` is pending. No production queue/worker invocation or training run has been submitted. This is a prerequisite dependency, not missing user authorization. Reporting implementation changes require a new tested snapshot and matching readiness before production admission; no passing gate may be relabeled. Four production terminals, exact campaign totals, ordinary endpoints and final comparison acceptance remain unproven.

T029/T030 are checked complete in tasks.md; T031 remains unchecked. The optional `/speckit.git.commit` after-implementation hook was not executed. Final `git diff --check` passed.
