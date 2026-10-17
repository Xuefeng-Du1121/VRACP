"""Trace-driven link lifetimes from SUMO (replaces Exp(mean R / v) in sysim.py).

Each simulated vehicle is bound to a SUMO vehicle that is inside the middle
6 km [7, 13] km of the 20-km road at the trace offset T0 (drawn per seed, so
all variants of one seed share T0 and the binding). When sysim builds a route
at time t, every hop gets the residual lifetime of a physical endpoint pair
taken from the trace at T0 + t:
  * v2v: the bound vehicle and a neighbour drawn uniformly from the vehicles
    within R (either direction); lifetime = time until their distance > R.
  * rsu: the bound vehicle if it is within R of an RSU, otherwise a covered
    vehicle drawn uniformly within R of it (the last relay before the RSU;
    the nearest covered vehicle if none is within R); lifetime = time until
    it leaves the disk of that RSU.
  * uav: the bound vehicle and the UAV of the nearest 1-km segment, if within
    R in 3-D; lifetime = time until the 3-D distance > R. UAVs fly at 150 m and
    patrol their segment back and forth at 50 km/h (random phase per seed). If
    no UAV is in range the hop is served by a vehicle relay (v2v lifetime).
RSUs sit 10 m off the road every 1 km (x = 500 + 1000 j), R = 300 m, as in
sysim.py. Lifetimes are right-censored at CAP = 300 s and floored at 100 ms
(one fractional step), which only matters for links already at the range edge.
"""
import os
import numpy as np

R = 300.0
CAP = 300.0
FLOOR = 0.1
UAV_H = 150.0
UAV_V = 50.0 / 3.6
RSU_Y = -10.0
SEG = 1000.0
MID = (7000.0, 13000.0)
K_TRACES = 5
HERE = os.path.dirname(os.path.abspath(__file__))


class Mobility:
    def __init__(self, speed_kmh, seed, N):
        # one of the K independent SUMO traces of this speed, chosen by seed
        d = np.load(os.path.join(HERE, "sumo", f"v{int(speed_kmh)}",
                                 f"s{seed % K_TRACES}", "trace.npz"))
        self.x, self.y, self.dt = d["x"], d["y"], float(d["dt"])
        self.n_t = self.x.shape[1]
        self.rng = np.random.default_rng(seed)
        span = self.n_t * self.dt - CAP - 70.0
        self.T0 = float(self.rng.uniform(0, span))
        self.uav_phase = self.rng.uniform(0, 2 * SEG / UAV_V, 32)   # one per segment
        k0 = self._k(0.0)[0]
        mid = np.flatnonzero((self.x[:, k0] > MID[0]) & (self.x[:, k0] < MID[1]))
        self.bind = list(self.rng.choice(mid, N, replace=False))
        self.log = dict(v2v=[], rsu=[], uav=[], uav_fallback=0, isolated=0, rebind=0)

    # ---------------------------------------------------------------- helpers
    def _k(self, t_ms):
        s = self.T0 + t_ms / 1e3
        k = int(s / self.dt)
        return min(k, self.n_t - 1), s / self.dt - k

    def _present(self, k):
        return np.flatnonzero(~np.isnan(self.x[:, k]))

    def _anchor(self, v, k):
        a = self.bind[v]
        if np.isnan(self.x[a, k]):            # left the road: rebind
            p = self._present(k)
            mid = p[(self.x[p, k] > MID[0]) & (self.x[p, k] < MID[1])]
            a = self.bind[v] = int(self.rng.choice(mid if len(mid) else p))
            self.log["rebind"] += 1
        return a

    def _uav_xy(self, j, s):
        """Ground position of the UAV of segment j at trace time s (seconds)."""
        u = ((s + self.uav_phase[j]) * UAV_V) % (2 * SEG)
        return j * SEG + np.where(u < SEG, u, 2 * SEG - u), 0.0

    def _rsu(self, xv):
        j = np.round((xv - 500.0) / SEG)
        return 500.0 + SEG * j

    def _life(self, k, frac, dist):
        """dist(ks) -> distances at step indices ks; residual lifetime in ms."""
        ks = np.arange(k, min(k + int(CAP / self.dt) + 2, self.n_t))
        d = dist(ks)
        bad = np.isnan(d) | (d > R)
        if not bad[1:].any():
            return CAP * 1e3
        i = 1 + int(np.argmax(bad[1:]))
        d0, d1 = d[i - 1], d[i]
        f = 1.0 if np.isnan(d1) or np.isnan(d0) or d1 == d0 else (R - d0) / (d1 - d0)
        life = (i - 1 + float(np.clip(f, 0, 1)) - frac) * self.dt
        return max(min(life, CAP), FLOOR) * 1e3

    # ------------------------------------------------------------- interface
    def covered(self, v, t_ms):
        k, _ = self._k(t_ms)
        a = self._anchor(v, k)
        xa, ya = self.x[a, k], self.y[a, k]
        return np.hypot(xa - self._rsu(xa), ya - RSU_Y) <= R

    def life(self, kind, v, t_ms):
        k, frac = self._k(t_ms)
        a = self._anchor(v, k)
        X, Y = self.x, self.y
        if kind == "uav":
            s0 = self.T0 + t_ms / 1e3
            j = int(X[a, k] // SEG)
            ux, uy = self._uav_xy(j, s0)
            if np.sqrt((X[a, k] - ux) ** 2 + (Y[a, k] - uy) ** 2 + UAV_H ** 2) <= R:
                def dist(ks):
                    u = self._uav_xy(j, ks * self.dt)[0]
                    return np.sqrt((X[a, ks] - u) ** 2 + Y[a, ks] ** 2 + UAV_H ** 2)
                L = self._life(k, frac, dist)
                self.log["uav"].append(L)
                return L
            self.log["uav_fallback"] += 1
            kind = "v2v"
        if kind == "rsu":
            p = self._present(k)
            dx = X[p, k] - self._rsu(X[p, k])
            cov = p[np.hypot(dx, Y[p, k] - RSU_Y) <= R]
            if a in set(cov.tolist()):
                c = a
            else:                              # last relay: a covered vehicle within R
                near = cov[np.hypot(X[cov, k] - X[a, k], Y[cov, k] - Y[a, k]) <= R]
                c = (int(self.rng.choice(near)) if len(near)
                     else int(cov[np.argmin(np.abs(X[cov, k] - X[a, k]))]))
            rx = self._rsu(X[c, k])
            L = self._life(k, frac, lambda ks: np.hypot(X[c, ks] - rx, Y[c, ks] - RSU_Y))
            self.log["rsu"].append(L)
            return L
        p = self._present(k)
        dd = np.hypot(X[p, k] - X[a, k], Y[p, k] - Y[a, k])
        nb = p[(dd <= R) & (p != a)]
        if not len(nb):
            self.log["isolated"] += 1
            return FLOOR * 1e3
        b = int(self.rng.choice(nb))
        L = self._life(k, frac, lambda ks: np.hypot(X[a, ks] - X[b, ks], Y[a, ks] - Y[b, ks]))
        self.log["v2v"].append(L)
        return L

    def summary(self):
        g = self.log
        out = {f"life_{k}": float(np.mean(g[k])) / 1e3 if g[k] else float("nan")
               for k in ("v2v", "rsu", "uav")}
        n_uav = len(g["uav"]) + g["uav_fallback"]
        out["uav_fallback"] = 100 * g["uav_fallback"] / max(n_uav, 1)
        out["isolated"] = g["isolated"]
        out["rebind"] = g["rebind"]
        return out
