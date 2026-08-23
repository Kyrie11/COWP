from __future__ import annotations

from cowp.waymax_eval.baselines import ablation_for_method


def test_ablation_switches_exist():
    assert ablation_for_method('cowp_wo_option_preservation')['use_option_preservation'] is False
    assert ablation_for_method('cowp_wo_witness_rejection')['use_hard_witness_rejection'] is False


def test_sharded_state_generator_filters_before_materialization(monkeypatch):
    from cowp.waymax_eval import dataloader

    raw_records = [f"record-{i}".encode() for i in range(6)]
    parsed_calls = []
    built_calls = []

    monkeypatch.setattr(
        dataloader,
        "iter_tfexample_records_sharded",
        lambda path, shard_index, num_shards: ((i, raw) for i, raw in enumerate(raw_records) if i % num_shards == shard_index),
    )

    def fake_parse(raw):
        parsed_calls.append(raw)
        return {"raw": raw}

    monkeypatch.setattr(dataloader, "parse_tfexample", fake_parse)
    monkeypatch.setattr(dataloader, "decode_parsed_tfexample", lambda parsed: parsed)

    def fake_build(example, include_sdc_paths=True, time_key="all"):
        built_calls.append(example["raw"])
        return f"state:{example['raw'].decode()}"

    monkeypatch.setattr(dataloader, "simulator_state_from_womd_dict", fake_build)
    rows = list(
        dataloader.waymax_state_generator_sharded(
            {"womd": {"validation_tfexample_glob": "unused"}},
            split="validation",
            tfexample_glob="unused",
            shard_index=1,
            num_shards=2,
        )
    )

    assert rows == [
        (1, "state:record-1"),
        (3, "state:record-3"),
        (5, "state:record-5"),
    ]
    assert parsed_calls == [raw_records[1], raw_records[3], raw_records[5]]
    assert built_calls == [raw_records[1], raw_records[3], raw_records[5]]


def test_waymax_at_shard_syntax_expands_to_tfrecord_files(tmp_path):
    from cowp.data.parse_tfexample import resolve_glob_patterns

    base = tmp_path / "validation_tfexample.tfrecord"
    expected = []
    for idx in range(3):
        path = tmp_path / f"validation_tfexample.tfrecord-{idx:05d}-of-00003"
        path.write_bytes(b"")
        expected.append(str(path))

    assert resolve_glob_patterns(f"{base}@3") == expected


def test_closed_loop_prefilter_preserves_global_scenario_index(monkeypatch):
    from cowp.waymax_eval import dataloader, rollout

    class State:
        def __init__(self, name, done=False):
            self.name = name
            self.done = done
            self.num_objects = 4

    class Env:
        def reset(self, state):
            return state

        def step(self, state, action):
            return State(state.name, done=True)

    monkeypatch.setattr(
        dataloader,
        "waymax_state_generator_sharded",
        lambda *args, **kwargs: iter([(1, State("one")), (3, State("three"))]),
    )
    monkeypatch.setattr(rollout, "_make_waymax_environment", lambda **kwargs: Env())
    seen = []

    def policy(state, *, step, scenario_index):
        seen.append((state.name, step, scenario_index))
        return object()

    outputs = rollout.waymax_closed_loop_rollout(
        {},
        policy,
        num_scenarios=2,
        horizon_steps=5,
        progress=False,
        shard_index=1,
        num_shards=2,
        prefilter_shards=True,
        jit_env=False,
        status_every=0,
    )

    assert seen == [("one", 0, 1), ("three", 0, 3)]
    assert [row["steps"] for row in outputs] == [1, 1]


def test_closed_loop_exact_ids_are_filtered_before_state_build_and_keep_global_indices(monkeypatch):
    from cowp.waymax_eval import dataloader, rollout

    class State:
        def __init__(self, name, done=False):
            self.name = name
            self.done = done
            self.num_objects = 4

    class Env:
        def reset(self, state):
            return state

        def step(self, state, action):
            return State(state.name, done=True)

    requested_sets = []

    def fake_exact_generator(*args, **kwargs):
        requested_sets.append(set(args[1]))
        # Simulate dataset scan order rather than allowlist order.
        for sid in ("scene-c", "scene-a"):
            if sid in args[1]:
                yield sid, State(sid)

    monkeypatch.setattr(dataloader, "waymax_state_generator_for_sids", fake_exact_generator)
    monkeypatch.setattr(rollout, "_make_waymax_environment", lambda **kwargs: Env())
    seen = []

    def policy(state, *, step, scenario_index):
        seen.append((state.name, scenario_index))
        return object()

    outputs = rollout.waymax_closed_loop_rollout(
        {},
        policy,
        scenario_ids=["scene-a", "scene-b", "scene-c", "scene-d"],
        horizon_steps=2,
        progress=False,
        shard_index=0,
        num_shards=2,
        prefilter_shards=True,
        jit_env=False,
        status_every=0,
    )

    assert requested_sets == [{"scene-a", "scene-c"}]
    assert seen == [("scene-c", 2), ("scene-a", 0)]
    assert [row["scenario_id"] for row in outputs] == ["scene-c", "scene-a"]


def test_closed_loop_exact_ids_fail_when_requested_id_is_missing(monkeypatch):
    import pytest
    from cowp.waymax_eval import dataloader, rollout

    class State:
        done = True
        num_objects = 4

    class Env:
        def reset(self, state):
            return state

        def step(self, state, action):
            return state

    monkeypatch.setattr(
        dataloader,
        "waymax_state_generator_for_sids",
        lambda *args, **kwargs: iter([("scene-a", State())]),
    )
    monkeypatch.setattr(rollout, "_make_waymax_environment", lambda **kwargs: Env())

    with pytest.raises(RuntimeError, match="resolved 1/2"):
        rollout.waymax_closed_loop_rollout(
            {},
            lambda state, **kwargs: object(),
            scenario_ids=["scene-a", "scene-b"],
            horizon_steps=1,
            progress=False,
            jit_env=False,
            status_every=0,
        )
