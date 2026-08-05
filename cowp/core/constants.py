from __future__ import annotations

from enum import IntEnum


class ObjectType(IntEnum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


class PriorityRelation(IntEnum):
    UNKNOWN = 0
    EGO_PRIORITY = 1
    AGENT_PRIORITY = 2
    EQUAL_OR_NEGOTIATED = 3


class MacroType(IntEnum):
    KEEP_LANE = 0
    YIELD = 1
    CREEP = 2
    STOP_BEFORE_CONFLICT = 3
    MERGE_AHEAD = 4
    MERGE_BEHIND = 5
    LANE_CHANGE_LEFT = 6
    LANE_CHANGE_RIGHT = 7
    ACCELERATE_CROSS = 8
    DECELERATE_CROSS = 9
    LOGGED_EGO = 10
    NEUTRAL_EGO = 11
    PAD = 12


class ProposalSource(IntEnum):
    """Provenance of an ego proposal primitive.

    Macro type describes the semantic action, while proposal source records how
    the trajectory entered the bank.  Keeping these notions separate is useful
    for proposal-ceiling audits: two MERGE_BEHIND trajectories may come from the
    legacy ego-only timing lattice or from the interaction-conditioned envelope.
    """

    PAD = 0
    LOGGED = 1
    KEEP = 2
    ACCELERATE = 3
    YIELD = 4
    STOP = 5
    LEGACY_TIMING = 6
    ROBUST_BCTE = 7
    LANE_CHANGE = 8
    CREEP = 9
    NEUTRAL = 10
    TERMINAL = 11


class NaturalSource(IntEnum):
    OBS = 0
    NEU = 1
    PRIO = 2
    PAD = 3


class ResponseSource(IntEnum):
    PRED = 0
    OPT = 1
    EMG = 2
    PAD = 3


class MechanismToken(IntEnum):
    NONE = 0
    HB = 1
    AY = 2
    PA = 3
    GS = 4
    SR = 5
    OR = 6


TOKEN_NAMES = {
    MechanismToken.NONE: "NONE",
    MechanismToken.HB: "HB",
    MechanismToken.AY: "AY",
    MechanismToken.PA: "PA",
    MechanismToken.GS: "GS",
    MechanismToken.SR: "SR",
    MechanismToken.OR: "OR",
}
