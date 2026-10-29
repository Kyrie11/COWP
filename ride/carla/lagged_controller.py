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
        self.requests=0
        self.cap_binding_steps=0
        self.lag_mismatch_steps=0
        self.max_requested_brake=0.0
        self.max_applied_brake=0.0
        self.last_sample=None

    def step(self, newly_computed):
        """Delay only C's brake command, never lateral steering or throttle."""
        out=_clone(newly_computed)
        requested=float(newly_computed.brake)
        self.brake_buffer.append(requested)
        delayed=float(self.brake_buffer.popleft())
        applied=min(delayed,self.cap)
        out.brake=applied
        self.requests+=1
        self.cap_binding_steps+=int(delayed>self.cap+1e-6)
        self.lag_mismatch_steps+=int(abs(delayed-requested)>1e-6)
        self.max_requested_brake=max(self.max_requested_brake,requested)
        self.max_applied_brake=max(self.max_applied_brake,applied)
        self.last_sample={'requested_brake':requested, 'delayed_brake':delayed,
                          'applied_brake':applied,
                          'cap_actually_binding':delayed>self.cap+1e-6,
                          'lag_actually_changed_brake':abs(delayed-requested)>1e-6}
        return out

    def actuation_report(self):
        return {'requested_delay_ticks':self.ticks,'command_cap':self.cap,
                'command_samples':self.requests,
                'command_cap_binding_ticks':self.cap_binding_steps,
                'delay_changed_command_ticks':self.lag_mismatch_steps,
                'max_raw_requested_C_brake':self.max_requested_brake,
                'max_applied_C_brake':self.max_applied_brake,
                'effectiveness':'CAP_AND_DELAY' if self.cap_binding_steps and self.lag_mismatch_steps
                    else 'CAP_ONLY' if self.cap_binding_steps else
                         'DELAY_ONLY' if self.lag_mismatch_steps else 'NO_EFFECT'}
