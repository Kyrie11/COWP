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

# Deliberately exploratory heterogeneous controller compositions.
# A and B choose the first family; C independently chooses the other one.
HETEROGENEOUS_FAMILIES = ('mixed_b_tm_c_behavior','mixed_b_behavior_c_tm',
                          'mixed_b_tm_c_delayed_behavior')

def per_role_response_modes(archetype, family):
    roles=response_roles(archetype)
    if family in ('tm','behavior'):
        return {r:family for r in roles}
    if archetype!=CHAIN:
        raise ValueError('Mixed responder policies are only defined for chain_cut_in')
    if family in ('mixed_b_tm_c_behavior','mixed_b_tm_c_delayed_behavior'):
        return {'follower_a':'tm','follower_b':'tm','follower_c':'behavior'}
    if family=='mixed_b_behavior_c_tm':
        return {'follower_a':'behavior','follower_b':'behavior','follower_c':'tm'}
    raise ValueError('Unknown responder family: '+str(family))
