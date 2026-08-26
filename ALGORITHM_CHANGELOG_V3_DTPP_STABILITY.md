# V3 DTPP stability changelog

- Restore DTPP public default `variable_cost=False`.
- Standard DTPP training uses FP32; AMP becomes explicit opt-in.
- DTPP AdamW weight decay aligned to PyTorch public-source implicit default 0.01.
- Separate encoder/decoder grad clipping at 5.0 with non-finite error gate.
- Check loss, gradients and parameters before allowing corrupted updates to propagate.
- Restrict malformed-batch skipping to adapter data-contract exceptions.
- Explicit ego/neighbor validity masks for WOMD ego-frame stopped actors.
- Robust map padding mask; local x=0 is not padding.
- Explicit candidate-valid mask keeps stationary stop branches and rejects padded branches.
- Correct B x heads attention-mask expansion with `repeat_interleave`.
- Restore DTPP 10-step max-pooling path for ego-tree attention.
- Add DTPP score/weight magnitude diagnostics.
- PLUTO agent-dropout now updates validity mask.
- Evaluation rejects non-finite scores/trajectories and loads checkpoints strictly.
- Parallel launchers reap/report both child workers.
