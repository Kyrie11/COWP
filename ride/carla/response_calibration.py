"""Prospective responder speed-range M0 diagnostics on negative controls.

Not a risk outcome or human realism measure. Detects sustained controller
speed-limit drift that would confound cross-response-model comparisons.
"""
from __future__ import annotations


def sustained_overspeed(trace:list[dict],role:str,nominal_mps:float, *, multiplier:float=1.5,
                       start_s:float=1.5, min_duration_s:float=0.5) -> bool:
    if nominal_mps<=0:raise ValueError('nominal_mps must be positive')
    count=0
    for row in trace:
        if float(row.get('t',0)) < start_s:
            count=0;continue
        if float(row[f'{role}_speed_mps']) > nominal_mps*multiplier:
            count+=1
            if count>=max(1,int(round(min_duration_s/0.05))):return True
        else:count=0
    return False
