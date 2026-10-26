"""2D oriented-vehicle-box overlap check with optional 3D height gating.

Avoids the false positives from center-distance or circumscribing-disk checks.
CARLA sensor collision events remain the authoritative *executed* collision
measurement. These helpers are only an INITIAL-state validity check.
"""
from __future__ import annotations
import math


def rects_intersect(cx1,cy1,yaw1,length1,width1,cx2,cy2,yaw2,length2,width2,tol=0.):
    """SAT overlap of two xy OBBs (yaw in *radians*; full length/width)."""
    def directions(yaw):
        return ((math.cos(yaw),math.sin(yaw)),(-math.sin(yaw),math.cos(yaw)))
    a0,a1=directions(yaw1);b0,b1=directions(yaw2)
    dp=(float(cx2)-float(cx1),float(cy2)-float(cy1))
    for axis in (a0,a1,b0,b1):
        def dot(u,v):return u[0]*v[0]+u[1]*v[1]
        ra=.5*length1*abs(dot(a0,axis))+.5*width1*abs(dot(a1,axis))
        rb=.5*length2*abs(dot(b0,axis))+.5*width2*abs(dot(b1,axis))
        if abs(dot(dp,axis)) >= ra+rb+tol:return False
    return True


def carla_actor_overlap(actor_a, actor_b):
    """Query actual bounding-box geometry; CARLA types imported lazily."""
    ta=actor_a.get_transform();tb=actor_b.get_transform()
    ba=actor_a.bounding_box;bb=actor_b.bounding_box
    ca=ta.transform(ba.location);cb=tb.transform(bb.location)
    if abs(ca.z-cb.z)>=ba.extent.z+bb.extent.z:return False
    ya=math.radians(ta.rotation.yaw+ba.rotation.yaw)
    yb=math.radians(tb.rotation.yaw+bb.rotation.yaw)
    return rects_intersect(ca.x,ca.y,ya,2*ba.extent.x,2*ba.extent.y,
                           cb.x,cb.y,yb,2*bb.extent.x,2*bb.extent.y)
