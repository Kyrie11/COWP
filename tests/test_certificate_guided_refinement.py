from cowp_extensions.certificate_guided_refinement import (
    ConflictRegion,
    EgoState,
    ProtectedPairCertificate,
    RefinementConfig,
    generate_certificate_guided_refinements,
    solve_constant_acceleration,
)


def test_constant_acceleration_reaches_target():
    sol = solve_constant_acceleration(
        distance_m=20.0,
        initial_speed_mps=5.0,
        target_time_s=3.0,
        min_accel_mps2=-3.5,
        max_accel_mps2=2.5,
    )
    assert sol is not None
    a, v = sol
    assert abs(5.0 * 3.0 + 0.5 * a * 9.0 - 20.0) < 1e-8
    assert v >= 0.0


def test_premature_stop_is_rejected():
    sol = solve_constant_acceleration(
        distance_m=1.0,
        initial_speed_mps=2.0,
        target_time_s=10.0,
        min_accel_mps2=-3.5,
        max_accel_mps2=2.5,
    )
    assert sol is None


def test_only_protected_deficits_generate():
    region = ConflictRegion("r0", 25.0, 3.5, 4.0, 4.6)
    protected = ProtectedPairCertificate(
        "a", "AgentPriority", 0.7, 0.2, 0.4, 0.1, (region,)
    )
    unprotected = ProtectedPairCertificate(
        "b", "EgoPriority", 0.9, 0.1, 0.9, 0.2, (region,)
    )
    proposals = generate_certificate_guided_refinements(
        ego_state=EgoState(7.0),
        certificates=[protected, unprotected],
        config=RefinementConfig(max_abs_jerk_proxy_mps3=20.0),
    )
    assert proposals
    assert {p.agent_id for p in proposals} == {"a"}
