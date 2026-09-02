# Near-contact external baselines

This package adds near-contact external baselines on top of the OC-RAP grouped candidate-prefix dataset.

## Implemented methods

- `marc_lite`: MARC-style multipolicy and risk-aware contingency planner. The OC-RAP semantic candidate lattice is treated as the ego policy set; latent roots are policy-conditioned critical scenarios; root divergence/common-prefix length is used as the dynamic branch-point proxy; selection is policy-level then candidate-level.
- `racp_lite`: RACP-style Bayesian belief contingency planner. Root probabilities are priors, branch recovery likelihoods update a posterior belief, and the cost combines shared-plan utility, posterior contingent value, and worst discounted risk.
- `expected_risk_filter`, `cvar_risk_filter`, `dro_cvar_filter`: safety filters over signed root-conditioned collision/recovery losses. The DR-CVaR variant adds a Wasserstein ambiguity penalty to the empirical CVaR loss.
- `predictive_safety_filter`: predictive safety / CBF backup filter. Candidate 0 is the proposed nominal input; it is kept if certified. Otherwise the filter chooses the smallest candidate-lattice modification that satisfies branch-wise backup and CBF-like recovery-barrier constraints.
- `oracle_recovery_filter`: strict branch-wise existential recovery baseline. A prefix is admitted only if every valid latent root has at least one valid recovery option above threshold.
- `gameformer_lite`: learned level-k GameFormer adapter from the safe-regime implementation; use the near-contact train/val/test splits.

## One-command run

```bash
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export WOMD_VAL_INTERACTIVE=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord
export RUN=runs/near_contact_external_baselines
export NUM_GPUS=2
bash scripts/run_near_contact_external_baselines.sh
```

The script writes offline grouped-candidate metrics and true Waymax receding-horizon closed-loop metrics to `$RUN`.

## Notes on training

MARC, RACP, the risk filters, PSF/CBF, and oracle recovery are optimization/filter baselines, not learned policy networks. Their `train-baseline` step validates the grouped dataset and writes a deterministic config summary. GameFormer is trained normally and then loaded for closed-loop evaluation.
