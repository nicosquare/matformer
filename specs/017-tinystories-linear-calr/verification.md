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
