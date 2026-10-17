"""RDRP link-break predictor (Sec. IV-B) and its hit/false-alarm measurement on
the SUMO traces (Sec. VI-C).

Predictor (Sec. IV-B, Algorithm 2). Every slot (100 ms) each endpoint's
reported position p, velocity v and acceleration a are extrapolated over the
prediction horizon Delta = 300 ms as p + v Delta + a Delta^2 / 2; the link
survival probability is psi = Pr(||p_i - p_j|| <= R) under Gaussian report
errors, and a break is flagged when psi < psi_trig.

Report errors (assumptions, swept in the sensitivity study):
    sigma_p  GNSS position error per axis (m),          default 1.5
    sigma_v  reported-speed error per axis (m/s),        default 0.5
    sigma_a  reported-acceleration error per axis (m/s^2), default 0.5
RSUs are static and exact; UAVs report like vehicles.

Ground truth: the true endpoint distance first exceeds R (3-D for the UAV), or
an endpoint leaves the road. A break is a HIT if it is flagged at least one
slot (100 ms) and at most Delta before it occurs, i.e. early enough for a
make-before-break switch. A flag is FALSE if no break follows within Delta.
Links are the same as in mobility.py: V2V = a vehicle in the middle 6 km and a
neighbour within R; RSU = a covered vehicle and its RSU; UAV = a vehicle and
the UAV of its segment within 3-D range. Each link is followed for up to
W_FOLLOW seconds; links that do not break within it count only for false
alarms.

"""
import json
import os
import sys

import numpy as np
from math import erf, sqrt

import mobility as MB

SLOT = 0.1                      # s
DELTA = 0.3                     # s, prediction horizon (3 slots)
LEAD = int(round(DELTA / SLOT))
W_FOLLOW = 120.0                # s followed per link
N_LINKS = 600                   # per kind and trace
SPEEDS = [20, 60, 120]
R = MB.R
HERE = os.path.dirname(os.path.abspath(__file__))
_ERF = np.vectorize(erf, otypes=[float])
PHI = lambda z: 0.5 * (1.0 + _ERF(np.asarray(z, float) / sqrt(2.0)))


def load(speed, s):
    d = np.load(os.path.join(HERE, "sumo", "v%d" % speed, "s%d" % s, "trace.npz"))
    return d["x"].astype(float), d["y"].astype(float), float(d["dt"])


def upsample(xs, dt):
    """Positions of one vehicle, trace step dt -> SLOT, linear in between;
    also the exact (piecewise-constant) velocity and acceleration."""
    n = len(xs)
    tt = np.arange(0, (n - 1) * dt + 1e-9, SLOT)
    k = np.minimum((tt / dt).astype(int), n - 2)
    f = tt / dt - k
    p = xs[k] * (1 - f) + xs[k + 1] * f
    v_step = np.diff(xs) / dt
    v = v_step[k]
    a_step = np.concatenate([[0.0], np.diff(v_step) / dt])
    a = a_step[k]
    return p, v, a


class Trace:
    def __init__(self, speed, s, rng):
        self.x, self.y, self.dt = load(speed, s)
        self.T = self.x.shape[1]
        self.rng = rng
        self.uav_phase = rng.uniform(0, 2 * MB.SEG / MB.UAV_V, 32)
        self.cache = {}

    def track(self, i):
        if i not in self.cache:
            px, vx, ax = upsample(self.x[i], self.dt)
            py, vy, ay = upsample(self.y[i], self.dt)
            self.cache[i] = (px, py, vx, vy, ax, ay)
        return self.cache[i]

    def uav(self, j, n_slots, s0):
        t = (s0 + np.arange(n_slots)) * SLOT
        u = ((t + self.uav_phase[j]) * MB.UAV_V) % (2 * MB.SEG)
        x = j * MB.SEG + np.where(u < MB.SEG, u, 2 * MB.SEG - u)
        vx = np.where(u < MB.SEG, MB.UAV_V, -MB.UAV_V)
        return x, np.zeros_like(x), vx, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)


def evaluate_link(A, B, h, sp, sv, sa, trig, rng, n_mobile):
    """A, B: (px, py, vx, vy, ax, ay) arrays over the followed window (B may be
    static). h: height offset (UAV). Returns (break slot or None, flags)."""
    px, py = A[0], A[1]
    qx, qy = B[0], B[1]
    d_true = np.sqrt((px - qx) ** 2 + (py - qy) ** 2 + h * h)
    bad = np.isnan(d_true) | (d_true > R)
    brk = int(np.argmax(bad)) if bad.any() else None
    L = brk if brk is not None else len(d_true)
    if L == 0:                                 # already out of range: skipped by the caller
        return brk, np.zeros(0, bool)

    def report(T, mobile):
        x, y, vx, vy, ax, ay = [np.asarray(c[:L], float) for c in T]
        if not mobile:
            return x, y, vx, vy, ax, ay
        n = lambda s: rng.normal(0, s, L)
        return x + n(sp), y + n(sp), vx + n(sv), vy + n(sv), ax + n(sa), ay + n(sa)

    ra = report(A, True)
    rb = report(B, n_mobile == 2)
    D = DELTA
    hx = (ra[0] + ra[2] * D + 0.5 * ra[4] * D * D) - (rb[0] + rb[2] * D + 0.5 * rb[4] * D * D)
    hy = (ra[1] + ra[3] * D + 0.5 * ra[5] * D * D) - (rb[1] + rb[3] * D + 0.5 * rb[5] * D * D)
    d_hat = np.sqrt(hx * hx + hy * hy + h * h)
    sig = sqrt(n_mobile * (sp ** 2 + (D * sv) ** 2 + (0.5 * D * D * sa) ** 2))
    psi = PHI((R - d_hat) / sig)
    return brk, psi < trig


def episodes(idx):
    """Number of maximal runs of consecutive slot indices."""
    return 0 if len(idx) == 0 else 1 + int(np.sum(np.diff(idx) > 1))


def score(brk, flags, min_lead=1):
    """hit (or None if no break), flagged slots, false flagged slots, false
    warning episodes (a run of consecutive false flags counts once)."""
    idx = np.flatnonzero(flags)
    if brk is None:
        return None, len(idx), len(idx), episodes(idx)
    lead = brk - idx                          # slots ahead of the break
    hit = bool(np.any((lead >= min_lead) & (lead <= LEAD)))
    fidx = idx[lead > LEAD]
    return hit, len(idx), len(fidx), episodes(fidx)


def run(speed, sp=1.5, sv=0.5, sa=0.5, trig=0.5, seed=0, n_links=N_LINKS, min_lead=1):
    out = {k: dict(breaks=0, hits=0, flags=0, false=0, false_ep=0, link_s=0.0) for k in ["v2v", "rsu", "uav"]}
    W = int(W_FOLLOW / SLOT)
    for s in range(MB.K_TRACES):
        rng = np.random.default_rng(1000 * seed + 17 * s + speed)
        tr = Trace(speed, s, rng)
        n_slot_total = int((tr.T - 1) * tr.dt / SLOT)
        for kind in ["v2v", "rsu", "uav"]:
            got = 0
            tries = 0
            while got < n_links and tries < 20 * n_links:
                tries += 1
                s0 = int(rng.integers(0, n_slot_total - W - 2))
                k0 = int(s0 * SLOT / tr.dt)
                pres = np.flatnonzero(~np.isnan(tr.x[:, k0]))
                mid = pres[(tr.x[pres, k0] > MB.MID[0]) & (tr.x[pres, k0] < MB.MID[1])]
                if len(mid) == 0:
                    continue
                i = int(rng.choice(mid))
                A = [c[s0:s0 + W] for c in tr.track(i)]
                if len(A[0]) < W or np.isnan(A[0][0]):
                    continue
                if kind == "v2v":
                    d0 = np.hypot(tr.x[pres, k0] - tr.x[i, k0], tr.y[pres, k0] - tr.y[i, k0])
                    nb = pres[(d0 <= R) & (pres != i)]
                    if len(nb) == 0:
                        continue
                    j = int(rng.choice(nb))
                    B = [c[s0:s0 + W] for c in tr.track(j)]
                    if len(B[0]) < W:
                        continue
                    brk, fl = evaluate_link(A, B, 0.0, sp, sv, sa, trig, rng, 2)
                elif kind == "rsu":
                    xr = MB.SEG * np.round((A[0][0] - 500.0) / MB.SEG) + 500.0
                    if np.hypot(A[0][0] - xr, A[1][0] - MB.RSU_Y) > R:
                        continue
                    z = np.zeros(W)
                    B = [z + xr, z + MB.RSU_Y, z, z, z, z]
                    brk, fl = evaluate_link(A, B, 0.0, sp, sv, sa, trig, rng, 1)
                else:
                    jseg = int(A[0][0] // MB.SEG)
                    B = tr.uav(jseg, W, s0)
                    if np.sqrt((A[0][0] - B[0][0]) ** 2 + A[1][0] ** 2 + MB.UAV_H ** 2) > R:
                        continue
                    brk, fl = evaluate_link(A, B, MB.UAV_H, sp, sv, sa, trig, rng, 2)
                if brk is not None and brk < 1:
                    continue
                hit, nf, nfalse, nep = score(brk, fl, min_lead)
                o = out[kind]
                o["flags"] += nf
                o["false"] += nfalse
                o["false_ep"] += nep
                o["link_s"] += len(fl) * SLOT
                if hit is not None:
                    o["breaks"] += 1
                    o["hits"] += int(hit)
                got += 1
    for o in out.values():
        o["hit_rate"] = o["hits"] / max(o["breaks"], 1)
        o["false_per_flag"] = o["false"] / max(o["flags"], 1)
        o["false_alarms_per_link_hour"] = o["false_ep"] / max(o["link_s"], 1e-9) * 3600
    return out
