from ocrap.data.split import scenario_split


def test_scenario_split_is_stable():
    assert scenario_split("abc") == scenario_split("abc")
