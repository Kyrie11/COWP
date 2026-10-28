"""Frozen third-party attribution for three- and four-car CARLA experiments.

The four-car primary pair is B--C, NOT A--B; the rear-most driver C has no
communication channel with the AV. All counts use exact role identities.
"""

CHAIN = 'chain_cut_in'


def all_roles(archetype):
    return ('ego', 'follower_a', 'follower_b', 'follower_c') if archetype == CHAIN else (
        'ego', 'follower_a', 'follower_b')


def response_roles(archetype):
    return all_roles(archetype)[1:]


def primary_pair(archetype):
    return ('follower_b', 'follower_c') if archetype == CHAIN else ('follower_a', 'follower_b')


def nominal_speed(role, *, archetype, cruise_mps, rear_speed_delta_mps=0.0,
                  chain_c_speed_delta_mps=0.0):
    val = float(cruise_mps)
    if role == 'follower_b' and archetype in ('cut_in', 'support_gap', CHAIN):
        return val + float(rear_speed_delta_mps)
    if role == 'follower_c' and archetype == CHAIN:
        return val + float(chain_c_speed_delta_mps)
    return val
