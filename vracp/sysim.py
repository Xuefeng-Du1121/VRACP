"""System-level simulator of the closed VRACP loop (Secs. VI-C to VI-H: speed
sweep, resource actuation, ablation, trust attacks, scalability).
All parameters are declared here a priori.

Topology
  * 6-km corridor with N_r = 6 RSUs (1 km apart); each RSU anchors one
    cluster and one 802.11p collision domain (10 MHz, 6-Mbps PHY, 1000-B
    frames). N vehicles are assigned round-robin to the 6 clusters.
  * Channel goodput per collision domain: Bianchi saturation goodput at the
    number of contenders n = vehicles in the cluster + RSU (log-linear
    interpolation through the values of Section ref_bounds at n = 1/16/64),
    multiplied by (1 - PER). Rayleigh block fading: PER = 1 - exp(-g_th / g).
    Default: fixed 6 Mbps (QPSK 1/2, g_th = 5 dB). Table VII: SNR-adaptive
    MCS over {6, 12, 27} Mbps (g_th = 5/11/20 dB), see pick_mcs.
  * 70% of vehicles are inside RSU coverage (1 hop); the others reach their
    RSU over 2 or 3 hops through vehicle or UAV relays of the same cluster.
  * Link lifetime ~ Exp(mean R / v), R = 300 m, v = vehicle speed
    (UAV hops: 50 km/h); RSU hop: Exp(2R / v). mobility="sumo" replaces
    these draws by residual lifetimes of real endpoint pairs from SUMO traces
    (mobility.py, sumo_gen.py); geo_hops=True additionally takes the 1-hop
    case from the trace (vehicle within R of an RSU) instead of P_COVER.

Tasks (Table III)
  * Poisson arrivals, lambda_total = 20 tasks/s, spread over honest vehicles.
    (c_r, Gamma_r) uniform over {1e8, 1e9} x {200, 500, 1000} ms; data
    volume B_r per experiment. Vehicle CPU 2 GHz, RSU 5 GHz (FIFO).
  * A task runs locally or is offloaded to the cluster RSU. Offloading
    transmits B_r over each hop of the route (FIFO on the cluster channel),
    then computes at the RSU; the small result returns in tau_min per hop.

Adversary: f Byzantine vehicles act as relays and drop 50% of the task
transfers / region reports they forward. A dropped transfer is detected
after RTO = 200 ms and retransmitted if the deadline can still be met.

Modules and interfaces (switched off by the ablation variants)
  * AMLC: two-tier PBFT (local per cluster, then representatives), GCS
    admission, local commit for own-region state. Without AMLC: flat PBFT.
    HierHotStuff: two-tier HotStuff without GCS admission.
  * Trust: GCS-v2 scores (gcs.py) from observed forwarding (F), validity
    (V = 1) and on-time ratio (T). Used for admission and, through the
    A->R interface, for relay eligibility (GCS >= theta_route).
  * Predict (RDRP): a link break is predicted with probability 0.9 under the
    exponential model, and under SUMO mobility with the hit rate measured
    for that speed and link kind (exp_predictor.py, predictor_rates.py);
    a predicted break switches the route to a prepared backup; otherwise the break is detected
    after 300 ms of missed beacons plus one discovery round trip; a transfer
    in progress over a broken hop fails and is restarted.
  * ITAC (async): decides on arrival from the cluster's current state,
    picking the option with the earliest estimated finish; the feasibility
    projection rejects a task that no option can finish by its deadline.
    R->I: the estimate uses the actual route (hop count); without it a
    1-hop route is assumed. Sync (w/o Async, DDPG baselines): decisions
    are taken at barriers every P = max(100 ms, 2 * max cross delay) from the
    state snapshot of the previous barrier. DDPG baselines have no
    feasibility projection.
  * I->A: the AMLC local-tier timeout is extended by the measured channel
    queueing delay; without it a fixed 6 tau_min + 50 ms timeout fires a
    view change when the local round is slowed by data-plane contention.

Metrics follow Eqs. delivery_metrics, queue_backlog, tp and PCC.
Consensus: rho_delay = 20, tau_min = 5 ms, Gamma_c = 500 ms, epoch 1 s.
QB: per-cluster channel backlog / Q_max (Q_max = 4000 kb), sampled each 100 ms.
"""
import heapq, bisect
import numpy as np
import predictor_rates as PR
from des import Net
from consensus import pbft_round, hotstuff_round
from gcs import GCS, score

M = 6
TAU_MIN = 5.0
RHO = 20
GAMMA_C = 500.0
EPOCH = 1000.0
SLOT = 100.0
R_RANGE = 300.0
UAV_KMH = 50.0
P_COVER = 0.7
LAMBDA = 20.0
C_SET = [1e8, 1e9]
# Gamma_r mix. 600 ms (not 500) for the middle class: local execution of
# c_r=1e9 on a 2-GHz vehicle takes exactly 500 ms, so a 500-ms deadline would
# sit exactly on the completion time and any queueing delay would flip the
# outcome, making DSR a knife-edge statistic rather than a measure of control.
G_SET = [200.0, 600.0, 1000.0]
F_VEH, F_RSU = 2e9, 5e9
BYZ_DROP = 0.5
RTO = 200.0
P_PRED = 0.9
BEACON = 300.0
Q_MAX = 4000e3
OBJ_PER_CL = 10
TURNOVER = 0.3
MISS = 0.02

FULL = dict(cons="AMLC", trust=True, a2r=True, predict=True, ctrl="ITAC",
            sync=False, r2i=True, i2a=True)
VARIANTS = {
    "Full VRACP": {},
    "w/o AMLC": dict(cons="PBFT"),
    "w/o Trust": dict(trust=False, a2r=False),
    "w/o Predict": dict(predict=False),
    "w/o Async": dict(sync=True),
    "w/o A->R": dict(a2r=False),
    "w/o R->I": dict(r2i=False),
    "w/o I->A": dict(i2a=False),
    "w/o Feedback (all)": dict(a2r=False, r2i=False, i2a=False),
    "Stchd.-Naive": dict(cons="PBFT", trust=False, a2r=False, predict=False,
                         ctrl="DDPG", sync=True, r2i=False, i2a=False),
    "Stchd.-Cons": dict(a2r=False, predict=False, ctrl="DDPG", sync=True,
                        r2i=False, i2a=False),
    "Stchd.-Two": dict(ctrl="DDPG", sync=True, r2i=False, i2a=False),
    "HierHotStuff": dict(cons="HierHS", trust=False, a2r=False, r2i=False, i2a=False),
}


def cfg_of(name):
    c = dict(FULL)
    c.update(VARIANTS[name])
    return c


# 802.11p (10 MHz) MCS set used in Section ref_bounds: rate (Mbps) ->
# Bianchi goodput at n = 1/16/64 and SNR threshold (dB).
MCS = {6: ([4.61, 3.53, 2.86], 5.0),      # QPSK 1/2
       12: ([7.70, 6.09, 4.95], 11.0),    # 16-QAM 1/2
       27: ([12.2, 10.2, 8.31], 20.0)}    # 64-QAM 3/4


def bianchi(n, rate=6):
    xs = np.log2([1, 16, 64])
    return float(np.interp(np.log2(max(n, 1)), xs, MCS[rate][0]))


def per_of(rate, snr_db):
    """Per-attempt frame error under Rayleigh block fading."""
    return 1 - np.exp(-10 ** (MCS[rate][1] / 10) / 10 ** (snr_db / 10))


def pick_mcs(snr_db, n):
    """SNR-adaptive MCS (Table VII only): the rate that maximises expected
    goodput Bianchi(n, rate) * (1 - PER), as throughput-maximising rate
    adaptation does. Fixed a priori; not tuned to results."""
    return max(MCS, key=lambda r: bianchi(n, r) * (1 - per_of(r, snr_db)))


class Sim:
    def __init__(self, variant, N, f, speed_kmh, B_kb, snr_db, seed, dur=60000.0, mcs="fixed",
                 drop=BYZ_DROP, cons_byz=True, badmouth=False, n_obs=8,
                 obs_agg="mean", stealth=False, mobility="exp", geo_hops=False):
        self.c = cfg_of(variant)
        self.N, self.f, self.dur = N, f, dur
        self.B = B_kb * 1e3
        self.rng = np.random.default_rng(seed)
        self.r_task = np.random.default_rng(seed + 1)      # common across variants
        self.r_byz = np.random.default_rng(seed + 2)
        self.cl = np.arange(N) % M
        # AMLC assumes f_k < z_k / 3 per cluster: Byzantine vehicles are
        # spread over clusters (round-robin over a random cluster order)
        order = self.rng.permutation(M)
        byz = []
        for j in range(f):
            k = order[j % M]
            pool = [i for i in range(N) if self.cl[i] == k and i not in byz]
            byz.append(int(self.rng.choice(pool)))
        self.byz = set(byz)
        self.honest = [i for i in range(N) if i not in self.byz]
        # A1 (selective dropping, Table gcs_attack): cons_byz=False makes the
        # Byzantine vehicles follow the consensus protocol and only drop
        # forwarded data at rate `drop`
        self.drop = drop
        self.cons_byz = cons_byz
        # A2: Byzantine observers withhold acknowledgements for honest targets,
        # so each node's forwarding evidence is the mean over n_obs sampled
        # observers rather than a global count.
        self.badmouth = badmouth
        self.n_obs = n_obs
        # how per-observer forwarding evidence is aggregated: "mean" (the
        # design evaluated in the paper) or "median" (robust to a minority of
        # badmouthing observers)
        self.obs_agg = obs_agg
        # stealth: the badmouthers forward and vote correctly, so nothing they
        # do about themselves is detectable; they attack only by withholding
        # acknowledgements for honest targets.
        self.stealth = stealth
        self.pend_ev = [None] * N
        self.cons_honest = set(self.honest) if cons_byz else set(range(N))
        n_c = [int(np.sum(self.cl == k)) + 1 for k in range(M)]
        self.rate = [pick_mcs(snr_db, n) if mcs == "adaptive" else 6 for n in n_c]
        self.G = [bianchi(n, r) * 1e3 * (1 - per_of(r, snr_db))
                  for n, r in zip(n_c, self.rate)]                     # bits/ms
        v = speed_kmh / 3.6
        self.life_v2v = R_RANGE / v * 1e3
        self.life_uav = R_RANGE / (UAV_KMH / 3.6) * 1e3
        self.life_rsu = 2 * R_RANGE / v * 1e3
        self.mob = None
        self.speed_kmh = speed_kmh
        if mobility == "sumo":
            from mobility import Mobility
            self.mob = Mobility(speed_kmh, seed + 4, N)     # shared by variants
        self.geo_hops = geo_hops and self.mob is not None
        self.ch = [[] for _ in range(M)]        # busy intervals per channel
        self.ch_free = np.zeros(M)               # FIFO tail per channel
        self.rsu_free = np.zeros(M)
        self.cpu_free = np.zeros(N)
        self.gcs = GCS(N, init=np.full(N, 0.95))
        self.ep = dict(fin=np.zeros(N), fout=np.zeros(N))
        self.routes = {}
        self.net = Net(self.cl, TAU_MIN, RHO, np.random.default_rng(seed + 3))
        self.maxd = self.net.base.max()
        self.P = max(SLOT, 2 * self.maxd)
        self.snap = (self.ch_free.copy(), self.rsu_free.copy(), self.cpu_free.copy(), 0.0)
        self.st = dict(tasks=0, ok=0, lat=[], sent=0, deliv=0, bits=0.0, qb=[], pcc=[],
                       n_loc=0, n_off=0, n_rej=0, n_aband=0, rep_sent=0, rep_ok=0,
                       stale_trust=[], hon_no_route=[], hon_excl=[], slack=[],
                       byz_frac=[], byz_frac_worst=[], tier_viol=[])

    # ------------------------------------------------------------ routing
    def _hop_life(self, kind, v, t):
        if self.mob is not None:
            return self.mob.life(kind, v, t)
        m = {"rsu": self.life_rsu, "v2v": self.life_v2v, "uav": self.life_uav}[kind]
        return self.rng.exponential(m)

    def _relays(self, v, k):
        pool = [i for i in range(self.N) if self.cl[i] == self.cl[v] and i != v]
        if self.c["a2r"]:
            ok = self.gcs.routable()
            pool = [i for i in pool if ok[i]] or pool
        return [int(self.rng.choice(pool)) for _ in range(k)]

    def _new_route(self, v, t):
        if self.geo_hops:
            H = 1 if self.mob.covered(v, t) else int(self.rng.integers(2, 4))
        else:
            H = 1 if self.rng.random() < P_COVER else int(self.rng.integers(2, 4))
        kinds = ["rsu"] + [("uav" if self.rng.random() < 0.5 else "v2v") for _ in range(H - 1)]
        return dict(H=H, relays=self._relays(v, H - 1), kinds=kinds,
                    brk=[t + self._hop_life(k, v, t) for k in kinds], up=t)

    def _p_pred(self, kind):
        # measured hit rate of the Sec. IV-B predictor under SUMO mobility
        return PR.hit(self.speed_kmh, kind) if self.mob is not None else P_PRED

    def route(self, v, t):
        r = self.routes.get(v)
        if r is None:
            r = self.routes[v] = self._new_route(v, t)
        # relays that became ineligible are replaced at the next use (A->R)
        if self.c["a2r"]:
            ok = self.gcs.routable()
            if any(not ok[x] for x in r["relays"]):
                r = self.routes[v] = self._new_route(v, max(t, r["up"]))
        while min(r["brk"]) <= t:
            tb = min(r["brk"])
            kb = r["kinds"][int(np.argmin(r["brk"]))]
            if self.c["predict"] and self.rng.random() < self._p_pred(kb):
                r = self._new_route(v, tb)
            else:
                rtt = 2 * r["H"] * (TAU_MIN + 8000 / self.G[self.cl[v]])
                r = self._new_route(v, tb + BEACON + rtt)
            self.routes[v] = r
        return r

    # ------------------------------------------------------------ channel
    def _occupy(self, k, t, dur):
        st = max(t, self.ch_free[k])
        self.ch_free[k] = st + dur
        return st + dur

    def transfer(self, v, t):
        """Send B over the current route; returns (arrival time or None, end time)."""
        r = self.route(v, t)
        t = max(t, r["up"])
        k = self.cl[v]
        dur = self.B / self.G[k]
        for h in range(r["H"]):
            t_end = self._occupy(k, t, dur)
            if r["brk"][h] < t_end:                    # hop broke mid-transfer
                if not (self.c["predict"] and self.rng.random() < self._p_pred(r["kinds"][h])):
                    return None, t_end
            t = t_end
            if h < r["H"] - 1:
                x = r["relays"][h]
                self.ep["fin"][x] += 1
                if x in self.byz and self.r_byz.random() < self.drop:
                    return None, t
                self.ep["fout"][x] += 1
        return t + TAU_MIN, t

    # ------------------------------------------------------------ control
    def estimate(self, v, t, state):
        ch_free, rsu_free, cpu_free, _ = state
        k = self.cl[v]
        loc = max(t, cpu_free[v]) + self.c_r / F_VEH * 1e3
        H = self.route(v, t)["H"] if self.c["r2i"] else 1
        tx = max(t, ch_free[k]) + H * self.B / self.G[k]
        off = max(tx, rsu_free[k]) + self.c_r / F_RSU * 1e3 + H * TAU_MIN
        return loc, off

    def task(self, t_arr, v, c_r, gam):
        self.st["tasks"] += 1
        dl = t_arr + gam
        self.c_r = c_r
        if self.c["sync"]:
            t = (np.floor(t_arr / self.P) + 1) * self.P
            state = self.snap
        else:
            t = t_arr
            state = (self.ch_free, self.rsu_free, self.cpu_free, t)
        loc, off = self.estimate(v, t, state)
        if self.c["ctrl"] == "ITAC" and min(loc, off) > dl:
            self.st["n_rej"] += 1
            return                                            # rejected
        if loc <= off:
            self.st["n_loc"] += 1
            st = max(t, self.cpu_free[v])
            self.cpu_free[v] = st + c_r / F_VEH * 1e3
            self._done(t_arr, self.cpu_free[v], dl)
            return
        k = self.cl[v]
        while True:
            self.st["sent"] += 1
            arr, t_end = self.transfer(v, t)
            if arr is not None:
                break
            t = t_end + RTO
            if self.c["ctrl"] == "ITAC" and t + self.B / self.G[k] > dl:
                self.st["n_aband"] += 1
                return                                        # abandoned
            if t > dl + 2000.0:
                self.st["n_aband"] += 1
                return
        self.st["deliv"] += 1
        self.st["n_off"] += 1
        self.st["bits"] += self.B
        st = max(arr, self.rsu_free[k])
        self.rsu_free[k] = st + c_r / F_RSU * 1e3
        self._done(t_arr, self.rsu_free[k] + self.routes[v]["H"] * TAU_MIN, dl)

    def _done(self, t_arr, t_fin, dl):
        self.st["lat"].append(t_fin - t_arr)
        self.st["ok"] += t_fin <= dl
        # deadline slack, for the knife-edge sensitivity check
        self.st["slack"].append(dl - t_fin)

    # ------------------------------------------------------------ consensus
    def _flat(self, fn, members, t0, vc, e=0):
        k = e
        lead, t = members[k % len(members)], t0
        while True:
            r = fn(self.net, members, self.cons_honest, lead, t)
            if r:
                return r
            t += vc
            k += 1
            lead = members[k % len(members)]

    def _hier(self, members, e, t0, fn, qdelay):
        local, reps = {}, []
        base_to = 6 * TAU_MIN + 50.0
        to = base_to + (qdelay if self.c["i2a"] else 0.0)
        for k in range(M):
            mem = [i for i in members if self.cl[i] == k]
            if not mem:
                continue
            s = e % len(mem)
            mem = mem[s:] + mem[:s]
            t = t0
            for _ in range(len(mem)):
                if mem[0] not in self.byz or not self.cons_byz:
                    break
                mem = mem[1:] + mem[:1]
                t += to
            r = fn(self.net, mem, self.cons_honest, mem[0], t) if len(mem) > 1 else {mem[0]: t}
            if not r:                       # quorum unreachable this epoch
                r = {i: np.inf for i in mem}
            # data-plane contention slows the local round; a timeout that
            # ignores it fires a spurious view change (I->A interface)
            if qdelay > 0:
                r = {i: x + qdelay for i, x in r.items()}
                if not self.c["i2a"] and qdelay > base_to - 3 * TAU_MIN:
                    r = {i: x + to for i, x in r.items()}
            local.update(r)
            reps.append((mem[0], mem))
        ids = [x for x, _ in reps]
        t_rep = max(local[x] for x in ids)
        g = fn(self.net, ids, set(ids), ids[0], t_rep) if np.isfinite(t_rep) else {}
        glob = {}
        for x, mem in reps:
            if x not in g:
                glob.update({i: np.inf for i in mem})
                continue
            d = self.net.phase({x: g[x]}, {x: mem}, 256)[x]
            d[x] = g[x]
            glob.update(d)
        return local, glob

    def epoch(self, e, t0, truth, ref, view):
        self.net.advance(t0)
        qdelay = float(np.mean(np.maximum(self.ch_free - t0, 0.0)))
        self.net.bg = min(0.9, float(np.mean(np.clip(self.ch_free - (t0 - EPOCH), 0, EPOCH))) / EPOCH)
        cons = self.c["cons"]
        if cons == "PBFT":
            r = self._flat(pbft_round, list(range(self.N)), t0, 6 * self.maxd + 50.0, e)
            r = {i: x + qdelay for i, x in r.items()}
            arr = {i: {k: r.get(i, np.inf) for k in range(M)} for i in self.honest}
        else:
            members = (sorted(int(x) for x in np.flatnonzero(self.gcs.admitted()))
                       if self.c["trust"] else list(range(self.N)))
            fn = hotstuff_round if cons == "HierHS" else pbft_round
            local, glob = self._hier(members, e, t0, fn, qdelay)
            arr = {i: {k: (local.get(i, np.inf) if self.cl[i] == k else glob.get(i, np.inf))
                       for k in range(M)} for i in self.honest}
        # region reports travel from a random honest vehicle over its route
        rep_ok = {}
        for k in range(M):
            src = [i for i in self.honest if self.cl[i] == k]
            if not src:
                rep_ok[k] = False
                continue
            v = int(self.rng.choice(src))
            rt = self.route(v, t0)
            ok = rt["up"] <= t0
            for x in rt["relays"]:
                self.ep["fin"][x] += 1
                if x in self.byz and self.r_byz.random() < self.drop:
                    ok = False
                    break
                self.ep["fout"][x] += 1
            rep_ok[k] = ok
            self.st["sent"] += 1                 # region reports count in PDR
            self.st["rep_sent"] += 1
            self.st["rep_ok"] += ok
            self.st["deliv"] += ok
        # A->R uses *committed* trust evidence: cluster k's evidence is usable
        # only if k's own commit landed within Gamma_c. A late commit leaves the
        # previous (stale) trust view in force for that cluster's members.
        commit_ok = {}
        for k in range(M):
            own = [i for i in self.honest if self.cl[i] == k]
            commit_ok[k] = bool(own) and (arr[own[0]][k] - t0 <= GAMMA_C)
        tru = set().union(*truth.values())
        vals = []
        for i in self.honest:
            for k in range(M):
                if rep_ok[k] and arr[i][k] - t0 <= GAMMA_C:
                    view[i][k] = set(ref[k])
            o = set().union(*view[i].values())
            vals.append(len(o & tru) / len(o | tru))
        self.st["pcc"].append(np.mean(vals))
        if self.c["trust"]:
            s, hold = [], []
            # evidence that missed its commit deadline is not discarded: it is
            # carried forward and applied with the next successful commit, so a
            # persistently silent leader still accrues strikes.
            pend = self.pend_ev
            for i in range(self.N):
                F = self.ep["fout"][i] / self.ep["fin"][i] if self.ep["fin"][i] else 1.0
                if self.badmouth:
                    obs = self.rng.choice([x for x in range(self.N) if x != i],
                                          min(self.n_obs, self.N - 1), replace=False)
                    # each observer reports its own view of i's forwarding: an
                    # honest observer reports F, a badmouthing observer withholds
                    # its acknowledgements and so reports 0.
                    rep = [0.0 if x in self.byz else F for x in obs]
                    if self.obs_agg == "median":
                        # a median survives until more than half the sampled
                        # observers are malicious
                        F = float(np.median(rep))
                    else:
                        F = float(np.mean(rep))
                # Byzantine members withhold votes: observed on-time ratio 0.
                # A *stealthy* badmouther instead behaves correctly on every
                # dimension that is measured about itself (it forwards and votes
                # normally) and attacks only by withholding acknowledgements for
                # others, so its own score stays indistinguishable from honest.
                byz_detectable = (i in self.byz and self.cons_byz
                                  and self.c["cons"] != "PBFT"
                                  and not self.stealth)
                T = 0.0 if byz_detectable else 0.98
                if i in self.byz and self.stealth:
                    F = 1.0                       # forwards correctly itself
                sc = score(F, 1.0, T, 1.0)
                if commit_ok[self.cl[i]]:
                    # apply the worst pending evidence together with this epoch
                    sc = min(sc, pend[i]) if pend[i] is not None else sc
                    pend[i] = None
                else:
                    pend[i] = sc if pend[i] is None else min(pend[i], sc)
                s.append(sc)
                hold.append(not commit_ok[self.cl[i]])
            self.gcs.update(s, hold=hold)
            self.st["stale_trust"].append(float(np.mean(hold)))
            rt, ad = self.gcs.routable(), self.gcs.admitted()
            self.st["hon_no_route"].append(float(np.mean(~rt[self.honest])))
            self.st["hon_excl"].append(float(np.mean(~ad[self.honest])))
            # --- BFT premise audit -------------------------------------
            # Theorem per-tier safety needs f_j < |G_j|/3 inside every
            # admitted cluster committee. Excluding honest nodes shrinks the
            # denominator, so badmouthing can push the Byzantine share over
            # 1/3 even without the attackers doing anything detectable.
            adm = [i for i in range(self.N) if ad[i]]
            if adm:
                self.st["byz_frac"].append(
                    sum(1 for i in adm if i in self.byz) / len(adm))
            worst, viol = 0.0, 0
            for k in range(M):
                mem = [i for i in adm if self.cl[i] == k]
                if not mem:
                    continue
                fr = sum(1 for i in mem if i in self.byz) / len(mem)
                worst = max(worst, fr)
                # quorum intersection fails once f_j >= |G_j|/3
                viol += int(sum(1 for i in mem if i in self.byz) * 3 >= len(mem))
            self.st["byz_frac_worst"].append(worst)
            self.st["tier_viol"].append(viol)
        self.ep["fin"][:] = 0
        self.ep["fout"][:] = 0

    # ------------------------------------------------------------ main loop
    def run(self):
        g = self.r_task
        n_t = g.poisson(LAMBDA * self.dur / 1e3)
        arr = np.sort(g.uniform(0, self.dur, n_t))
        src = g.choice(self.honest, n_t)
        cs = g.choice(C_SET, n_t)
        gs = g.choice(G_SET, n_t)
        orng = np.random.default_rng(int(g.integers(1 << 30)))
        nid = 0
        truth = {}
        for k in range(M):
            truth[k] = set(range(nid, nid + OBJ_PER_CL)); nid += OBJ_PER_CL
        view = {i: {k: set(truth[k]) for k in range(M)} for i in self.honest}
        next_e, e, next_b, next_q = 0.0, 0, 0.0, 0.0
        for t_arr, v, c_r, gam in zip(arr, src, cs, gs):
            while next_e <= t_arr:
                for k in range(M):
                    kk = int(round(TURNOVER * OBJ_PER_CL))
                    gone = set(orng.choice(sorted(truth[k]), kk, replace=False))
                    truth[k] = (truth[k] - gone) | set(range(nid, nid + kk)); nid += kk
                ref = {k: {o for o in truth[k] if orng.random() > MISS} for k in range(M)}
                self.epoch(e, next_e, truth, ref, view)
                e += 1
                next_e += EPOCH
            while next_b <= t_arr:
                self.snap = (self.ch_free.copy(), self.rsu_free.copy(), self.cpu_free.copy(), next_b)
                next_b += self.P
            while next_q <= t_arr:
                bl = np.maximum(self.ch_free - next_q, 0.0) * np.array(self.G)
                self.st["qb"].append(min(1.0, float(bl.sum() / (M * Q_MAX))))
                next_q += SLOT
            self.task(t_arr, int(v), float(c_r), float(gam))
        s = self.st
        mob = self.mob.summary() if self.mob is not None else {}
        return dict(**mob,
            PCC=100 * float(np.mean(s["pcc"])),
            PDR=100 * s["deliv"] / max(s["sent"], 1),
            DSR=100 * s["ok"] / s["tasks"],
            tau=float(np.mean(s["lat"])) if s["lat"] else float("nan"),
            TP=s["bits"] / self.dur,
            QB=float(np.mean(s["qb"])),
            # offload accounting (audit): shares of all generated tasks
            off_ratio=100 * s["n_off"] / max(s["tasks"], 1),
            loc_ratio=100 * s["n_loc"] / max(s["tasks"], 1),
            rej_ratio=100 * s["n_rej"] / max(s["tasks"], 1),
            aband_ratio=100 * s["n_aband"] / max(s["tasks"], 1),
            task_sent=int(s["sent"] - s["rep_sent"]),
            task_deliv=int(s["deliv"] - s["rep_ok"]),
            rep_PDR=100 * s["rep_ok"] / max(s["rep_sent"], 1),
            task_PDR=100 * (s["deliv"] - s["rep_ok"]) / max(s["sent"] - s["rep_sent"], 1),
            stale_trust=100 * float(np.mean(s["stale_trust"])) if s["stale_trust"] else 0.0,
            hon_no_route=100 * float(np.mean(s["hon_no_route"])) if s["hon_no_route"] else 0.0,
            hon_excl=100 * float(np.mean(s["hon_excl"])) if s["hon_excl"] else 0.0,
            byz_frac=100 * float(np.mean(s["byz_frac"])) if s["byz_frac"] else 0.0,
            byz_frac_worst=100 * float(np.mean(s["byz_frac_worst"])) if s["byz_frac_worst"] else 0.0,
            tier_viol=float(np.mean(s["tier_viol"])) if s["tier_viol"] else 0.0,
            DSR_tol=100 * float(np.mean(np.asarray(s["slack"]) >= -10.0)) if s["slack"] else 0.0,
            edge=100 * float(np.mean(np.abs(np.asarray(s["slack"])) <= 5.0)) if s["slack"] else 0.0,
        )
