"""Exploratory delayed *brake-command* C policy for CARLA physics screening.

C's BehaviorAgent computes its normal command at each step. Apply *current*
steering and throttle, but delay only brake commands by an integral number of
physics steps. Delayed brake is additionally capped as a dimensionless control
fraction [0,1]. Neither modification is a validated human reaction-time model
or a guarantee of any particular deceleration.
"""
from __future__ import annotations
from collections import deque
import math


def _clone(control):
    copied=type(control)()
    for key in ('throttle','steer','brake','hand_brake','reverse','manual_gear_shift','gear'):
        if hasattr(control,key):setattr(copied,key,getattr(control,key))
    return copied


class DelayedCController:
    def __init__(self, initial_control, *, delay_s, brake_command_cap, dt=0.05):
        if (not math.isfinite(delay_s) or not 0<=delay_s<=1.5 or
            not math.isfinite(brake_command_cap) or not 0.15<=brake_command_cap<=1.0 or
            dt<=0 or not math.isfinite(dt)):
            raise ValueError('Invalid delayed C brake controller parameter')
        self.ticks=round(delay_s/dt)
        if abs(self.ticks*dt-delay_s)>1e-6:
            raise ValueError('Delay must be integer multiple of simulator step')
        self.cap=brake_command_cap
        self.brake_buffer=deque(float(initial_control.brake) for _ in range(self.ticks))

    def step(self, newly_computed):
        """Delay only C's brake command, never lateral steering or throttle."""
        out=_clone(newly_computed)
        self.brake_buffer.append(float(newly_computed.brake))
        out.brake=min(self.brake_buffer.popleft(),self.cap)
        return out
