# OC-RAP Patch Summary

This patch focuses on making the code suitable for the four-dataset training/evaluation protocol described in the README and diagnosed in the uploaded experiment results.

## Main fixes

1. **Mixed dataset geometry support**
   - `OCRAPSampleDataset` now pads heterogeneous root/option tensors to fixed model geometry.
   - Padded roots/options are marked invalid and excluded by masks in losses, OC-MERO, inference, and DRS.
   - This allows mixing natural/strict/post-contact sets with proof-artifact sets whose root/option counts differ.

2. **Checkpointing**
   - Training now writes:
     - `best.pt`
     - `latest.pt`
     - `checkpoints/epoch_XXXX.pt` for every epoch by default
   - Checkpoints include optimizer state and full model geometry.

3. **GPU-compatible inference with mixed geometry**
   - Model inference now pads each sample to the checkpoint geometry before moving tensors to the model device.

4. **Evaluation metric correctness**
   - `deployable_recovery_success` now accepts `root_valid` and ignores invalid/padded roots before probability normalization.
   - Out-of-range option indices are treated as failed roots instead of crashing.

5. **CLI override robustness**
   - `--set` and ablation flags work both before and after the subcommand.

6. **Smoke validation**
   - `PYTHONPATH=src pytest -q` passes: `30 passed`.
   - A small synthetic train/evaluate smoke run was executed successfully.
