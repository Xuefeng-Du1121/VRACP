"""Semantic-message simulator: TPR / COR / Stale / FOR (Sec. VI-F).

Parameters are fixed a priori (declared in the paper) and are NOT tuned to
reproduce any earlier number.

Topology / traffic
  * N=24 nodes, 4 clusters x 6; one shared 802.11p channel per cluster
    (collision domain), 6-Mbps PHY, 1000-B frames. Per-attempt channel time
    = 8000 b / 3.53 Mbps (Bianchi saturation goodput, 16 contenders) = 2.27 ms.
  * Per-attempt frame error rate p_loss (setting); MAC retry limit 7.
  * Every honest node generates perception objects at 10 Hz; 25% are
    critical (transition-zone / occluded). Deadline Gamma_r uniform in
    {200,500,1000} ms. 75% of objects are destined to another cluster.
  * Route hop count: normal U{1,2,3}, critical U{2,3,4}. A cross-cluster route
    splits its hops between source and destination channels; the RSU/UAV
    backhaul adds the one-way cross delay tau_min*rho**U(0.5,1) (tau_min=5 ms).
  * Mean link lifetime 15 s (normal) and 5 s (transition zone/UAV); a route
    of H hops breaks at rate H/lifetime. With MOBILITY = "sumo" a route
    breaks at the first of its hop lifetimes taken from SUMO traces
    (mobility.py; source vehicle as anchor): RSU hop + V2V relays for normal
    routes, RSU hop + UAV relays for critical (transition-zone) routes. The
    exponential draw is still made and discarded so that the hop-count and
    prediction draws stay aligned with the exp runs.

Mechanisms (on/off per variant; mapping follows the variant definitions in
the paper's Baseline/ablation paragraph)
  * Predict (RDRP link-survival prediction + make-before-break): a break is
    predicted (0.9 under the exponential model, measured rate under SUMO) and the route switches to a prepared
    backup without interruption. Otherwise the route is down for
    300 ms (3 missed 100-ms beacons) + one discovery round trip; objects
    generated meanwhile wait at the source.
  * Admission filter (AMLC/GCS, inside the consensus module): objects are
    accepted only if the source is admitted (GCS >= theta_adm); objects from
    non-admitted sources are dropped by the first honest relay.
  * Relay eligibility (A->R interface): relays must be route-eligible
    (GCS >= theta_route).
  * Feedback (ITAC deadline/path-budget feedback): channels serve
    earliest-deadline-first. Without it, channels are FIFO.
  * Stitched-Naive: independently combined modules (PBFT + AODV + DDPG) with
    no GCS admission filter and no relay eligibility, reactive route repair,
    FIFO channels. Stitched-Cons is the variant that adds trust filtering.
  * w/o Trust removes all GCS-based filtering; w/o Feedback keeps all three
    modules but removes the A->R and R->I interfaces.

Adversary
  * f Byzantine insiders start with GCS 0.95 (same as honest nodes).
    As relays they drop 50% of forwarded frames. As sources they inject
    forged critical objects at 10 Hz. All methods apply the validity check
    against the committed reference, which rejects forged objects in regions
    covered by honest observers (probability 0.5); forged objects in
    uncovered regions pass it.
  * GCS inputs per epoch (1 s): F = observed forwarding ratio; V = 1 for
    honest, share of forged objects passing the reference check for
    Byzantine sources; T = share of the node's transmissions whose channel
    sojourn is <= 50 ms; A = 1. Score and strike rule as in gcs.py.

Metrics follow Eqs. semantic_metrics/stale_for, per (object, consumer).
"""
import sys, json, heapq
import numpy as np
import predictor_rates as PR
from gcs import GCS, score

N, M = 24, 4
TAU_MIN = 5.0
SLOT_T = 2.27              # ms per transmission attempt
RETRY = 7
RATE_HZ = 10.0
P_CRIT = 0.25
DEADLINES = [200.0, 600.0, 1000.0]   # matches Gamma_r in Table sim_params
P_CROSS = 0.75
LIFE = {False: 15000.0, True: 5000.0}
P_PRED = 0.9
BEACON_DET = 300.0
BYZ_DROP = 0.5
P_COVER = 0.5
HOP_BUDGET = 50.0
EPOCH = 1000.0
EPOCHS = 60
MOBILITY, SPEED = "exp", 20.0      # set from the command line
SETTINGS = {"Mild": (5, 0.05, 2), "Default": (20, 0.10, 3), "Harsh": (40, 0.20, 6)}
VARIANTS = {  # admission filter, relay eligibility (A->R), predict, EDF (R->I)
    "Stchd.-Naive": (False, False, False, False),
    "w/o Trust": (False, False, True, True),
    "w/o Predict": (True, True, False, True),
    "w/o Feedback": (True, False, True, False),
    "VRACP": (True, True, True, True),
}


def one_run(variant, setting, seed):
    trust, elig, predict, feedback = VARIANTS[variant]
    rho, p_loss, f = SETTINGS[setting]
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(M), N // M)
    byz = set(int(x) for x in rng.choice(N, f, replace=False))
    cross = np.full((M, M), TAU_MIN)
    for a in range(M):
        for b in range(a + 1, M):
            cross[a, b] = cross[b, a] = TAU_MIN * rho ** rng.uniform(0.5, 1.0)
    i, j = np.unravel_index(np.argmax(cross), cross.shape)
    cross[i, j] = cross[j, i] = TAU_MIN * rho
    r_route = np.random.default_rng(seed + 1)
    r_mac = np.random.default_rng(seed + 2)
    r_byz = np.random.default_rng(seed + 3)
    gcs = GCS(N, init=np.full(N, 0.95))
    mob = None
    if MOBILITY == "sumo":
        from mobility import Mobility
        mob = Mobility(SPEED, seed + 5, N)

    ev, seq = [], [0]

    def push(t, kind, data):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, data))

    # objects: [src, crit, forged, t_gen, deadline_abs, hops(list of cl), cross_after, idx]
    q = [[] for _ in range(M)]          # channel queues
    busy = [False] * M
    stats = dict(ref=0, ref_ok=0, crit=0, crit_ok=0, recv=0, stale=0, forged=0, forged_acc=0)
    ep = dict(fw_in=np.zeros(N), fw_out=np.zeros(N), tx=np.zeros(N), tx_ok=np.zeros(N),
              fg=np.zeros(N), fg_pass=np.zeros(N))

    routes = {}
    # read-only trust audit: honest nodes below theta_route / theta_adm per epoch
    # (does not consume random numbers, so TPR/COR/Stale/FOR are unchanged)
    audit = dict(ne=0, no_route=0, no_adm=0, ever_no_route=np.zeros(N, bool),
                 ever_no_adm=np.zeros(N, bool), T_sum=np.zeros(N), T_n=np.zeros(N),
                 soj_max=0.0, soj_n=0, soj_gt=0, util=np.zeros(M), pool_empty=0)

    def new_route(src, dc, crit, t):
        H = int(r_route.integers(2, 5) if crit else r_route.integers(1, 4))
        brk = t + r_route.exponential(LIFE[crit] / H)
        kind = "v2v"
        if mob is not None:
            kinds = ["rsu"] + ["uav" if crit else "v2v"] * (H - 1)
            lives = [mob.life(k, src, t) for k in kinds]
            i = int(np.argmin(lives))
            brk, kind = t + lives[i], kinds[i]
        return [H, brk, t, kind]               # hops, next break, up-from, breaking hop kind

    def route_for(src, dc, crit, t):
        key = (src, dc, crit)
        r = routes.get(key) or new_route(src, dc, crit, t)
        while r[1] <= t:                       # process breaks up to t
            tb = r[1]
            p_pred = PR.hit(SPEED, r[3]) if mob is not None else P_PRED
            if predict and r_route.random() < p_pred:
                r = new_route(src, dc, crit, tb)
            else:
                H = r[0]
                rtt = 2 * (H * SLOT_T + (cross[cl[src], dc] if dc != cl[src] else 0))
                up = tb + BEACON_DET + rtt
                r = new_route(src, dc, crit, up)
                r[2] = up
        routes[key] = r
        return r

    def relay_pool():
        if elig:
            ok = gcs.routable()
            p = [k for k in range(N) if ok[k]]
            # under extreme congestion every node can lose route eligibility;
            # record the collapse and fall back to the unfiltered pool
            if not p:
                audit["pool_empty"] += 1
                return list(range(N))
            return p
        return list(range(N))

    def gen(t, src, crit, forged):
        sc = cl[src]
        dc = int(r_route.integers(0, M - 1))
        if r_route.random() < P_CROSS:
            dc = dc if dc < sc else dc + 1
        else:
            dc = sc
        r = route_for(src, dc, crit, t)
        H = r[0]
        t_send = max(t, r[2])
        dl = t + DEADLINES[int(r_route.integers(0, 3))]
        if dc != sc:
            h1 = (H + 1) // 2
            chans = [sc] * h1 + [dc] * (H - h1)
            cross_after = h1 - 1
        else:
            chans, cross_after = [sc] * H, -1
        pool = relay_pool()
        relays = [int(pool[int(r_route.integers(0, len(pool)))]) for _ in range(H - 1)]
        obj = dict(src=src, crit=crit, forged=forged, tg=t, dl=dl, chans=chans,
                   xa=cross_after, relays=relays, k=0, dc=dc)
        if not forged:
            stats["ref"] += 1
            stats["crit"] += crit
        else:
            stats["forged"] += 1
        push(t_send, "enq", obj)

    def enqueue(t, o):
        c = o["chans"][o["k"]]
        key = o["dl"] if feedback else t
        heapq.heappush(q[c], (key, seq[0], t, o)); seq[0] += 1
        if not busy[c]:
            start(t, c)

    def start(t, c):
        while q[c]:
            _, _, ta, o = heapq.heappop(q[c])
            att = 1
            while r_mac.random() < p_loss and att < RETRY:
                att += 1
            ok = att < RETRY or r_mac.random() >= p_loss
            busy[c] = True
            push(t + att * SLOT_T, "done", (c, o, ok, ta))
            return
        busy[c] = False

    def deliver(t, o):
        stats["recv"] += 1
        late = t > o["dl"]
        stats["stale"] += late
        adm = (not trust) or bool(gcs.admitted()[o["src"]])
        if o["forged"]:
            caught = r_byz.random() < P_COVER
            ep["fg"][o["src"]] += 1
            ep["fg_pass"][o["src"]] += (not caught)
            if adm and not caught:
                stats["forged_acc"] += 1
        elif adm and not late:
            stats["ref_ok"] += 1
            stats["crit_ok"] += o["crit"]

    # generation events (independent of variant -> common random numbers)
    g = np.random.default_rng(seed + 4)
    for n in range(N):
        t = g.uniform(0, 100.0)
        while t < EPOCHS * EPOCH:
            crit = g.random() < P_CRIT
            if n in byz:
                push(t, "gen", (n, True, True))
            else:
                push(t, "gen", (n, bool(crit), False))
            t += 1000.0 / RATE_HZ
    for e in range(1, EPOCHS + 1):
        push(e * EPOCH - 1e-6, "epoch", None)

    while ev:
        t, _, kind, d = heapq.heappop(ev)
        if kind == "gen":
            gen(t, *d)
        elif kind == "enq":
            o = d
            # first honest relay drops traffic of non-admitted sources
            if trust and o["k"] == 1 and not gcs.admitted()[o["src"]]:
                if o["forged"]:                # the relay still runs the check
                    ep["fg"][o["src"]] += 1
                    ep["fg_pass"][o["src"]] += r_byz.random() >= P_COVER
                continue
            enqueue(t, o)
        elif kind == "done":
            c, o, ok, ta = d
            tx_node = o["src"] if o["k"] == 0 else o["relays"][o["k"] - 1]
            ep["tx"][tx_node] += 1
            ep["tx_ok"][tx_node] += (t - ta) <= HOP_BUDGET
            if tx_node not in byz:                 # sojourn audit (honest only)
                audit["soj_max"] = max(audit["soj_max"], t - ta)
                audit["soj_n"] += 1
                audit["soj_gt"] += (t - ta) > HOP_BUDGET
            audit["util"][c] += t - ta
            start(t, c)
            if not ok:
                continue
            o["k"] += 1
            if o["k"] == len(o["chans"]):
                deliver(t, o)
                continue
            rl = o["relays"][o["k"] - 1]
            ep["fw_in"][rl] += 1
            if rl in byz and r_byz.random() < BYZ_DROP:
                continue
            ep["fw_out"][rl] += 1
            dt = cross[o["chans"][o["k"] - 1], o["chans"][o["k"]]] if o["k"] - 1 == o["xa"] else 0.0
            push(t + dt, "enq", o)
        elif kind == "epoch":
            s = []
            for n in range(N):
                F = ep["fw_out"][n] / ep["fw_in"][n] if ep["fw_in"][n] else 1.0
                V = ep["fg_pass"][n] / ep["fg"][n] if ep["fg"][n] else 1.0
                T = ep["tx_ok"][n] / ep["tx"][n] if ep["tx"][n] else 1.0
                s.append(score(F, V, T, 1.0))
            gcs.update(s)
            hon_e = [n for n in range(N) if n not in byz]
            rt, ad = gcs.routable(), gcs.admitted()
            audit["ne"] += len(hon_e)
            audit["no_route"] += int(np.sum(~rt[hon_e]))
            audit["no_adm"] += int(np.sum(~ad[hon_e]))
            audit["ever_no_route"] |= ~rt
            audit["ever_no_adm"] |= ~ad
            for n in hon_e:
                T = ep["tx_ok"][n] / ep["tx"][n] if ep["tx"][n] else 1.0
                audit["T_sum"][n] += T; audit["T_n"][n] += 1
            for k in ep:
                ep[k][:] = 0
    hon = [n for n in range(N) if n not in byz]
    return dict(
        TPR=100 * stats["ref_ok"] / stats["ref"],
        COR=100 * stats["crit_ok"] / stats["crit"],
        Stale=100 * stats["stale"] / max(stats["recv"], 1),
        FOR=100 * stats["forged_acc"] / max(stats["forged"], 1),
        honest_excluded=int(np.sum(~gcs.admitted()[hon])),
        # false-exclusion audit (honest nodes only)
        FER_route=100 * audit["no_route"] / max(audit["ne"], 1),
        FER_adm=100 * audit["no_adm"] / max(audit["ne"], 1),
        ever_route=100 * np.sum(audit["ever_no_route"][hon]) / len(hon),
        ever_adm=100 * np.sum(audit["ever_no_adm"][hon]) / len(hon),
        T_mean=float(np.mean(audit["T_sum"][hon] / np.maximum(audit["T_n"][hon], 1))),
        T_min=float(np.min(audit["T_sum"][hon] / np.maximum(audit["T_n"][hon], 1))),
        soj_max=audit["soj_max"],
        soj_gt=100 * audit["soj_gt"] / max(audit["soj_n"], 1),
        pool_empty=audit["pool_empty"],
    )
