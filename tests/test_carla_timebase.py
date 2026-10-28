from pathlib import Path


def test_each_recorded_control_step_has_one_carla_tick():
    """Guard the physical/logical time-base used by all CARLA evidence."""
    source = (Path(__file__).parents[1] / "ride" / "carla" / "run.py").read_text()
    loop = source.split("for step in range(round(args.duration_s/DT)):", 1)[1]
    loop = loop.split("# OBB is a sensor-health diagnostic only", 1)[0]
    assert loop.count("world.tick()") == 1


def test_manifest_declares_tick_accounting():
    source = (Path(__file__).parents[1] / "ride" / "carla" / "run.py").read_text()
    assert "physics_ticks_per_recorded_step':1" in source
