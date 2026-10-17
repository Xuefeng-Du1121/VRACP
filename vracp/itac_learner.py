"""Phase-1 ITAC learner: an actor--critic in place of the earliest-finish rule.

The system simulator (sysim.py) normally runs ITAC's feasibility-projected
*rule*: offload iff the estimated offload finish is earlier than the local
one, reject iff neither option meets the deadline. Here the placement
decision is taken by a learned stochastic policy instead, everything else
(mobility, channel, consensus, trust, routing, task stream) is unchanged.

Agents. One coordinator per cluster (M = 6), as in Algorithm 3.
State x (11 features, from the controller's view of the cluster: the live
  state, or the last barrier snapshot for the synchronous learner):
  bias, local and offload deadline slack / Gamma_r (clipped to [-3, 3]),
  route hop count / 3 (R->I), channel, RSU and own-CPU backlog (s),
  heavy-task flag, Gamma_r (s), B_r (Mb), and the estimated offload-minus-
  local finish gap / Gamma_r (the quantity the rule thresholds at zero).
Action: local (0) or offload (1). pi(offload|x) = sigmoid(theta . x).
Projection: the policy is restricted to the feasible options {estimated
  finish <= deadline}; with one feasible option it is taken, with none the
  task is rejected (the admission step of Sec. IV-C). "w/o projection"
  samples from both options and never rejects.
Reward: -(latency / Gamma_r) - lambda * miss, miss = deadline missed or
  transfer abandoned; dual update lambda <- [lambda + eta_l (miss - eps)]_+.
  An outcome is used only once the simulated clock passes its completion
  time (outcomes are associated with their decision when observed).
TD errors are clipped to +-DELTA_CLIP and lambda to [0, LAM_MAX].
Critic: linear V(x), TD(0) with discount GAMMA over the agent's decisions;
  actor: theta += alpha_a * delta * grad log pi (advantage actor--critic).
Asynchronous aggregation (Algorithm 3): every G_SYNC local updates an agent
  pushes its accumulated delta with its version stamp; the push reaches the
  aggregator after D_AGG ms, is discarded if the aggregator's version has
  advanced by more than TAU_MAX since the agent's last pull, and the agent
  pulls the aggregate D_AGG ms later. Nobody waits.
Synchronous learner: decisions at barriers from the barrier snapshot
  (sysim's sync mode, as for "w/o Async") and parameters merged at every
  barrier, i.e. the barrier-based global synchronization ITAC avoids.

Protocol: train for EPISODES 60-s episodes on training seeds 7000+e (disjoint
from the evaluation seeds); every EVAL_EVERY episodes evaluate the frozen
greedy policy (argmax over feasible options) on the evaluation seeds
5000..5009 used by every table and figure; repeat for LEARNER_SEEDS learner
initialisations. Default configuration (N=24, f=4, 20 km/h, B=1000 kb,
15 dB, SUMO mobility) at lambda = 20 and, near capacity, 100 tasks/s.

"""
import json
import sys

import numpy as np

import sysim as S

GAMMA = 0.9
ALPHA_A, ALPHA_C = 0.01, 0.02
DELTA_CLIP, LAM_MAX = 3.0, 5.0
ETA_L, EPS_MISS, LAM0 = 0.01, 0.02, 1.0
G_SYNC = 20                    # local updates between pushes
TAU_MAX = 8                    # staleness bound (aggregator versions)
D_AGG = S.RHO * S.TAU_MIN      # one-way delay to the regional aggregator (ms)
EPISODES = 150
EVAL_EVERY = 10
LEARNER_SEEDS = 5
EVAL_SEEDS = list(range(5000, 5010))
NF = 11
CFG = dict(N=24, f=4, v=20.0, B=1000.0, snr=15.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Learner:
    """Shared state of all agents: aggregator copy, local copies, buffers."""

    def __init__(self, seed, mode, projection=True):
        self.mode = mode                       # "async" | "sync"
        self.proj = projection
        self.rng = np.random.default_rng(seed)
        self.th_g = np.zeros(NF)
        self.ph_g = np.zeros(NF)
        self.v_g = 0
        self.lam = LAM0
        self.train = True
        self.stale = []
        self.dropped = 0
        self.reset_episode()

    def reset_episode(self):
        M = S.M
        self.th = [self.th_g.copy() for _ in range(M)]
        self.ph = [self.ph_g.copy() for _ in range(M)]
        self.ver = [self.v_g] * M
        self.dth = [np.zeros(NF) for _ in range(M)]
        self.dph = [np.zeros(NF) for _ in range(M)]
        self.n_upd = [0] * M
        self.buf = [[] for _ in range(M)]      # pending transitions per agent
        self.events = []                       # (time, kind, agent, payload)
        self.next_barrier = None

    # ---------------------------------------------------------- acting
    def act(self, k, x, feas):
        th = self.th[k]
        p1 = sigmoid(th @ x)
        if self.proj:
            if feas[0] and feas[1]:
                a = int(p1 > 0.5) if not self.train else int(self.rng.random() < p1)
                g = (a - p1) * x
            else:
                a, g = (0 if feas[0] else 1), np.zeros(NF)
        else:
            a = int(p1 > 0.5) if not self.train else int(self.rng.random() < p1)
            g = (a - p1) * x
        return a, g

    # ---------------------------------------------------------- learning
    def record(self, k, x, glog, r, t_obs, t_now):
        if not self.train:
            return
        b = self.buf[k]
        if b and b[-1]["xn"] is None:
            b[-1]["xn"] = x
        b.append(dict(x=x, g=glog, r=r, t=t_obs, xn=None))
        self.flush(t_now)

    def _update(self, k, tr):
        V = self.ph[k] @ tr["x"]
        Vn = self.ph[k] @ tr["xn"]
        delta = float(np.clip(tr["r"] + GAMMA * Vn - V, -DELTA_CLIP, DELTA_CLIP))
        dph = ALPHA_C * delta * tr["x"]
        dth = ALPHA_A * delta * tr["g"]
        if self.mode == "async":               # sync: applied only at the barrier
            self.ph[k] += dph
            self.th[k] += dth
        self.dph[k] += dph
        self.dth[k] += dth
        self.n_upd[k] += 1

    def flush(self, t_now):
        # deliver aggregator messages that have arrived
        self.events.sort(key=lambda e: e[0])
        while self.events and self.events[0][0] <= t_now:
            _, kind, k, pay = self.events.pop(0)
            if kind == "push":
                dth, dph, ver = pay
                s = self.v_g - ver
                self.stale.append(s)
                if s <= TAU_MAX:
                    self.th_g += dth
                    self.ph_g += dph
                    self.v_g += 1
                else:
                    self.dropped += 1
            else:                                          # pull
                self.th[k] = self.th_g.copy()
                self.ph[k] = self.ph_g.copy()
                self.ver[k] = self.v_g
        # apply the transitions whose outcome is observed by now
        for k in range(S.M):
            b = self.buf[k]
            while b and b[0]["xn"] is not None and b[0]["t"] <= t_now:
                self._update(k, b.pop(0))
                if self.mode == "async" and self.n_upd[k] % G_SYNC == 0:
                    self.events.append((t_now + D_AGG, "push", k,
                                        (self.dth[k].copy(), self.dph[k].copy(), self.ver[k])))
                    self.events.append((t_now + 2 * D_AGG, "pull", k, None))
                    self.dth[k][:] = 0
                    self.dph[k][:] = 0

    def barrier(self):
        """Synchronous learner: merge every agent's delta, everyone pulls."""
        if self.mode != "sync" or not self.train:
            return
        for k in range(S.M):
            self.th_g += self.dth[k]
            self.ph_g += self.dph[k]
            self.dth[k][:] = 0
            self.dph[k][:] = 0
        self.v_g += 1
        self.stale.append(0)
        for k in range(S.M):
            self.th[k] = self.th_g.copy()
            self.ph[k] = self.ph_g.copy()
            self.ver[k] = self.v_g

    def dual(self, miss):
        if self.train:
            self.lam = min(LAM_MAX, max(0.0, self.lam + ETA_L * (miss - EPS_MISS)))


class LearnSim(S.Sim):
    def __init__(self, *a, learner=None, **kw):
        super().__init__(*a, **kw)
        self.L = learner
        self.last_bar = 0.0

    def features(self, v, t, state, loc, off, dl, gam):
        ch_free, rsu_free, cpu_free = state[:3]
        k = self.cl[v]
        H = self.route(v, t)["H"] if self.c["r2i"] else 1
        return np.array([1.0,
                         np.clip((dl - loc) / gam, -3, 3), np.clip((dl - off) / gam, -3, 3),
                         H / 3.0,
                         max(ch_free[k] - t, 0.0) / 1e3, max(rsu_free[k] - t, 0.0) / 1e3,
                         max(cpu_free[v] - t, 0.0) / 1e3,
                         float(self.c_r >= 1e9), gam / 1e3, self.B / 4e6,
                         np.clip((off - loc) / gam, -3, 3)])

    def task(self, t_arr, v, c_r, gam):
        L = self.L
        self.st["tasks"] += 1
        dl = t_arr + gam
        self.c_r = c_r
        if self.c["sync"]:
            t = (np.floor(t_arr / self.P) + 1) * self.P
            state = self.snap
            if t > self.last_bar:                  # a barrier has passed
                L.barrier()
                self.last_bar = t
        else:
            t = t_arr
            state = (self.ch_free, self.rsu_free, self.cpu_free, t)
        L.flush(t)
        loc, off = self.estimate(v, t, state)
        feas = (loc <= dl, off <= dl)
        if L.proj and not (feas[0] or feas[1]):
            self.st["n_rej"] += 1
            return                                     # empty feasible set
        k = self.cl[v]
        x = self.features(v, t, state, loc, off, dl, gam)
        a, glog = L.act(k, x, feas)
        if a == 0:
            self.st["n_loc"] += 1
            st = max(t, self.cpu_free[v])
            self.cpu_free[v] = st + c_r / S.F_VEH * 1e3
            self._done(t_arr, self.cpu_free[v], dl)
            t_fin, miss = self.cpu_free[v], self.cpu_free[v] > dl
        else:
            t_fin, miss = self._offload(t_arr, t, v, c_r, dl)
        lat = (t_fin - t_arr) if t_fin is not None else 2 * gam
        miss = bool(miss) or t_fin is None
        L.dual(float(miss))
        L.record(k, x, glog, -lat / gam - L.lam * miss, t_fin if t_fin is not None else dl, t)

    def _offload(self, t_arr, t, v, c_r, dl):
        k = self.cl[v]
        proj = self.L.proj
        while True:
            self.st["sent"] += 1
            arr, t_end = self.transfer(v, t)
            if arr is not None:
                break
            t = t_end + S.RTO
            if (proj and t + self.B / self.G[k] > dl) or t > dl + 2000.0:
                self.st["n_aband"] += 1
                return None, True
        self.st["deliv"] += 1
        self.st["n_off"] += 1
        self.st["bits"] += self.B
        st = max(arr, self.rsu_free[k])
        self.rsu_free[k] = st + c_r / S.F_RSU * 1e3
        t_fin = self.rsu_free[k] + self.routes[v]["H"] * S.TAU_MIN
        self._done(t_arr, t_fin, dl)
        return t_fin, t_fin > dl


VARIANTS = {
    "AC-async": dict(mode="async", projection=True, sync=False),
    "AC-sync": dict(mode="sync", projection=True, sync=True),
    "AC-async w/o proj.": dict(mode="async", projection=False, sync=False),
}
METRICS = ["DSR", "tau", "TP", "PCC", "PDR", "off_ratio", "rej_ratio", "aband_ratio"]


def sim(seed, learner, sync, lam):
    S.LAMBDA = lam
    sm = LearnSim("Full VRACP", CFG["N"], CFG["f"], CFG["v"], CFG["B"], CFG["snr"], seed,
                  mobility="sumo", learner=learner)
    if sync:
        sm.c["sync"] = True
    learner.reset_episode()
    return sm.run()


def evaluate(L, sync, lam):
    L.train = False
    rs = [sim(s, L, sync, lam) for s in EVAL_SEEDS]
    L.train = True
    return {m: float(np.mean([r[m] for r in rs])) for m in METRICS}


def train_job(a):
    name, lseed, lam = a
    cfg = VARIANTS[name]
    L = Learner(10_000 + lseed, cfg["mode"], cfg["projection"])
    curve = [dict(ep=0, **evaluate(L, cfg["sync"], lam))]
    train_dsr = []
    for ep in range(1, EPISODES + 1):
        r = sim(7000 + 1000 * lseed + ep, L, cfg["sync"], lam)
        train_dsr.append(r["DSR"])
        if ep % EVAL_EVERY == 0:
            curve.append(dict(ep=ep, **evaluate(L, cfg["sync"], lam)))
    st = np.asarray(L.stale) if L.stale else np.zeros(1)
    return a, dict(curve=curve, train_dsr=train_dsr, theta=L.th_g.tolist(), lam=L.lam,
                   stale_mean=float(st.mean()), stale_max=int(st.max()),
                   stale_dropped=L.dropped, pushes=len(L.stale))
