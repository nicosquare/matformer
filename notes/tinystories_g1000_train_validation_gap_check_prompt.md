# Prompt: check the g1000 training-versus-validation gap

In `/home/ivo.navarrete/ElasticNN/matformer`, run a **read-only training-versus-validation loss check** for `g1000`. Compare the terminal checkpoints from these completed runs:

- Standalone: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/ST-g1000`
- Coverage-balanced linear: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-coverage-balanced-v1/runs/tinystories-coverage-balanced-v1-S1-s42`
- Coverage-balanced geometric: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-coverage-balanced-geometric-v1/runs/tinystories-coverage-balanced-geometric-v1-S1-s42`

Select a deterministic, fixed subset from the **optimizer-training data** and evaluate all three `g1000` terminal checkpoints on exactly the same examples. Use the repository's existing causal language-model loss and target-token-weighted aggregation. Record the selected example IDs, selection rule, and a hash of the selection. Use the saved ordinary-validation terminal endpoints for the validation comparison.

Verify checkpoint, dataset, model-width, and validation identities before comparing. Do not train, update checkpoints, or evaluate the sealed final holdout. Save a CSV/JSON report and a clear plot under a fresh directory in `/nfs-stor/ivo.navarrete/results/elasticnn`. Show each model's training-subset loss, validation loss, and its differences from the standalone on both sets. Explain what the pattern suggests about fitting versus generalization, without treating this single check as proof of causation.

If a GPU job is needed, submit one job using the project's Slurm settings, exclude `gpu-[05,50-51,54]`, and confirm that it runs on GPU. Complete the check and report the results; do not stop at a proposed plan.
