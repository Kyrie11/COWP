# Regime Visualization v51

This patch replaces the old global-best-baseline / Near+Contact-only qualitative
pipeline with a three-regime, all-main-baseline pipeline.

## Scientific contract

- **Safe**: select five high absolute-quality OC-RAP closed-loop examples under
  collision/off-road guards. Bounded NUP and low unnecessary intervention are
  primary; relative external performance is only a non-inferiority tie-breaker.
- **Near-Contact**: compare OC-RAP against every current main-table external
  baseline on TTC, clearance and critical-exposure effects. The strict tier
  requires material improvement against the *per-scene best* external method,
  material gains against a majority of external methods, and no regression
  against any external method.
- **Contact**: same all-baseline contract using post-contact terminal clearance,
  free-space AUC, clearance gain, overlap duration, escape/stable-stop,
  re-contact and dynamic-stability evidence.
- **Duration**: first require at least 5.0 s of WOMD future horizon. Only when
  fewer than five candidates survive does the entire regime fall back to 3.0 s.
- Scene selection is metric-only and happens before any selective render trace
  is generated. Selection JSON stores all-scene scores, tiers, per-baseline
  terms and the duration fallback decision.

These examples are intentionally marked as qualitative/post-hoc. Population
claims should come from the complete paired closed-loop test tables.

## Current main-table baseline set

Safe (6): `gameformer_lite`, `plantf`, `pluto`, `pdm_closed`, `pdm_hybrid`, `idm`.

Near (6): `marc_lite`, `racp_lite`, `robust_scenario_mpc`,
`predictive_safety_filter`, `dr_cvar_safety_filter`,
`conformal_predictive_safety_filter`.

Contact (6): `postimpact_mpc_lite`, `post_crash_braking`,
`postimpact_motion_tvlqr`, `post_collision_restoration`,
`compensatory_postimpact_mpc`, `robust_postimpact_control`.

The reporting names come from `ocrap.external_baselines.provenance`; they retain
"adapter/projection" wording where appropriate and must not be described as the
original authors' official implementations unless provenance says so.

## Video contract

For each selected target with `n` external baselines, generate exactly `n+3`
videos:

1. OC-RAP single-model video.
2. One single-model video for each external baseline.
3. OC-RAP **left** vs the per-scene best external baseline **right**.
4. OC-RAP **left** vs the per-scene worst external baseline **right**.

With the current 6/6/6 lineup, 5 scenes per regime produce 45 videos per regime
and 135 total.

All videos for the same scene use the same encoded duration, the same simulator
time samples, and (by default) one fixed world-frame camera computed over all
models. The renderer verifies that every trace starts at the selected
`target_time_index` and that `time_index` is consecutive. Early-terminated
rollouts hold the final state and label this explicitly.

The map shows oriented vehicle boxes, SDC trail, roadgraph, observed contact
anchor, off-road/overlap status and the *actual minimum oriented-box pair* used
for the clearance connector. Safe uses clearance + speed timeline; Near uses
clearance + TTC; Contact uses clearance + speed, while the header reports
post-contact terminal clearance, free-space AUC, overlap duration, re-contact,
stable-stop and escape.

## Main entry points

- `tools/select_regime_visualization_scenes.py`
- `tools/render_regime_visualization_videos.py`
- `scripts/build_external_comparison_artifacts.sh`
- `scripts/run_selected_regime_visualization_traces.sh`
- `scripts/build_regime_visualization_videos.sh`
- `visualization-commands-v51.txt`

The external Safe/Near/Contact runners now expose scene-detail environment
variables so full benchmark runs remain metric-only while selective reruns can
request full render traces without duplicating runner logic.
