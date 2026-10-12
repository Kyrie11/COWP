"""Conservative circle vs oriented vehicle-footprint collision geometry.

This remains a diagnostic geometry model: box yaw for a moving ego candidate is
estimated from its path tangent; no Waymax inverse-dynamics execution is assumed.
"""
from __future__ import annotations
import numpy as np


def tangent_yaws(xy, fallback):
    xy=np.asarray(xy,dtype=float)
    out=np.full(len(xy),float(fallback))
    if len(xy)<=1:return out
    d=np.diff(xy,axis=0)
    for i,v in enumerate(d, start=1):
        if float(np.linalg.norm(v))>0.02:out[i]=np.arctan2(v[1],v[0])
        else:out[i]=out[i-1]
    out[0]=out[1] if np.linalg.norm(d[0])>0.02 else fallback
    return out


def oriented_clear(xy1,xy2,fp1,fp2,margin=0.0):
    """True if oriented rectangles remain separated at EVERY sampled timestamp.

    fp=(length_m,width_m,initial_yaw_rad,follow_path_tangent).
    Separation uses the 4-axis SAT. Touching is treated as safe only at zero
    penetration; margin inflates each box by half the requested clearance.
    """
    a=np.asarray(xy1,float);b=np.asarray(xy2,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[-1]!=2:raise ValueError('bad path shapes')
    l1,w1,y1,follow1=fp1;l2,w2,y2,follow2=fp2
    for z in [l1,w1,l2,w2,margin]:
        if not np.isfinite(z) or z<0:raise ValueError('bad footprint dimension')
    yaw1=tangent_yaws(a,y1) if follow1 else np.full(len(a),y1)
    yaw2=tangent_yaws(b,y2) if follow2 else np.full(len(b),y2)
    u1=np.column_stack([np.cos(yaw1),np.sin(yaw1)])
    u2=np.column_stack([np.cos(yaw2),np.sin(yaw2)])
    v1=np.column_stack([-u1[:,1],u1[:,0]])
    v2=np.column_stack([-u2[:,1],u2[:,0]])
    delta=a-b
    half1=(float(l1)/2+margin/2,float(w1)/2+margin/2)
    half2=(float(l2)/2+margin/2,float(w2)/2+margin/2)
    separated=np.zeros(len(a),dtype=bool)
    for axis in [u1,v1,u2,v2]:
        dist=np.abs(np.einsum('nd,nd->n',delta,axis))
        r1=half1[0]*np.abs(np.einsum('nd,nd->n',u1,axis))+half1[1]*np.abs(np.einsum('nd,nd->n',v1,axis))
        r2=half2[0]*np.abs(np.einsum('nd,nd->n',u2,axis))+half2[1]*np.abs(np.einsum('nd,nd->n',v2,axis))
        separated|=dist>=r1+r2
    return bool(np.all(separated))


def circle_clear(xy1,xy2,r1,r2,margin=0.0):
    a=np.asarray(xy1,float);b=np.asarray(xy2,float)
    if a.shape!=b.shape:raise ValueError('path length mismatch')
    return bool(np.all(np.linalg.norm(a-b,axis=-1)>=r1+r2+margin))
