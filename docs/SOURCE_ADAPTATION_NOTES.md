# Source adaptation notes: GameFormer and BeTopNet

This note records how the uploaded upstream baseline source designs were adapted to the OC-RAP candidate-prefix dataset.

## GameFormer mapping

Observed upstream modules:

- `model/GameFormer.py`: top-level encoder-decoder structure;
- `Encoder`: ego/neighbor/map encoders plus Transformer context fusion;
- `InitialDecoder`: learned multimodal queries and first-stage GMM trajectory proposal;
- `InteractionDecoder`: iterative level-k reasoning;
- `FutureEncoder`: encodes previous-level trajectories before the next interaction step;
- `level_k_loss`: supervises every reasoning level.

OC-RAP adapter mapping:

- `GameFormerLevelK` encodes `ego_history` and `neighbor_history` with LSTM blocks and fuses scene tokens with a Transformer encoder.
- Learned mode embeddings generate a level-0 multimodal trajectory set per candidate.
- `GameFormerFutureEncoder` encodes previous-level predicted trajectories.
- Each level performs interaction self-attention over candidate futures and cross-attention to scene context before predicting the next multimodal trajectory set.
- Training uses deep trajectory imitation over every level and mode classification using the closest mode.
- Candidate logits combine the existing candidate-policy score with multimodal trajectory confidence.

## BeTopNet mapping

Observed upstream modules:

- `betop_decoder.py`: build topology layers, run topology reasoning, then decode trajectories/plans;
- `topo_decoder.py`: topology fusers and topology decoders;
- `topo_utils.py`: behavior braid and map braid topology target construction;
- `topo_attention.py`: topology-aware attention with selected topological neighbors/maps;
- `loss_utils.py`: binary focal top-k topology loss.

OC-RAP adapter mapping:

- `BeTopNetLite` builds actor-topology and map-topology feature tensors from OC-RAP candidate prefixes, nearby actors, and local map polylines.
- Actor topology labels approximate behavior braids by checking candidate prefix overlap/crossing with constant-velocity neighbor extrapolations.
- Map topology labels approximate map braids by checking prefix-to-polyline occupancy/crossing.
- Separate `TopoFuser` and binary topology decoders predict actor/map topology logits.
- Top-k topology indexing selects the most relevant actor and map topology tokens.
- Topology-aware attention injects the selected topology memories into each candidate token before final candidate scoring.
- Training uses binary focal top-k topology losses plus the same candidate choice supervision and metrics as the other baselines.
