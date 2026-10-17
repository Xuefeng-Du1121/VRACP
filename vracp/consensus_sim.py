"""Consensus simulator: robustness under delay heterogeneity (Sec. VI-B).

Manuscript setting: N=16, f/N=12.5% (f=2), traffic 1000 kb/s,
rho_delay in {1,5,10,20,30,40}.

Declared modelling choices (reported in the paper's setup):
  * 4 geographic clusters x 4 nodes; the 2 Byzantine replicas sit in distinct
    clusters and are never a cluster's initial leader slot.
  * intra-cluster one-way delay tau_min = 5 ms; cross-cluster pair delays
    tau_min * rho**U(0.5,1), largest pair = rho * tau_min; log-normal jitter
    sigma = 0.25.
  * 6-Mbps PHY; 1500-B proposals, 256-B votes; 1 ms signature verification
    (PBFT-Perf: MAC authenticators, 0.2 ms).
  * one consensus instance per 1-s epoch; rotating leader; a Byzantine leader
    is silent and triggers a view change after timeout 6*max_delay+50 ms
    (HotStuff: 14*max_delay+50 ms; AMLC/ML-PBFT local tier: 6*tau_min+50 ms).
  * Byzantine replicas do not vote. Forged objects are removed by the
    validity check, which all methods apply, so this experiment isolates
    timing effects.
  * tau_cons = mean over honest replicas of (commit time - proposal time).
    AMLC: local (intra-cluster) commit; all others: full commit.
  * commit deadline Gamma_c = 500 ms; DMR = Pr[tau_cons > Gamma_c] over
    (honest replica, epoch) pairs.
  * PCC (Eq. PCC): a replica refreshes the objects of region c only if the
    commit covering c reaches it within Gamma_c; otherwise it keeps its last
    (stale) view. 30% object turnover per epoch; honest sensing miss 2%.
  * AMLC also runs GCS-v2 admission; Byzantine replicas that withhold votes
    get forwarding score F=0.6 (<0.75), hence s_i=0, and are struck after
    3 epochs.
"""
import sys, json
import numpy as np
from des import Net
from consensus import pbft_round, hotstuff_round
from gcs import GCS, score

N, M, F = 16, 4, 2
TAU_MIN = 5.0
GAMMA_C = 500.0
EPOCH = 1000.0
EPOCHS = 60
OBJ_PER_CL = 10
TURNOVER = 0.3
MISS = 0.02
RHOS = [1, 5, 10, 20, 30, 40]
METHODS = ["PBFT", "ML-PBFT", "PBFT-Perf", "HotStuff", "AMLC"]
SIG_B = 256


def bg_util(traffic_kbps, phy=6.0, eff=0.75):
    return traffic_kbps / M / (phy * 1e3 * eff)


def flat(net, fn, honest, leader, t0, vc):
    t, lead = t0, leader
    while True:
        r = fn(net, list(range(N)), honest, lead, t)
        if r:
            return r
        t += vc
        lead = (lead + 1) % N


def hier(net, cl, members, honest, byz, e, t0):
    """Local PBFT per cluster -> representative PBFT -> intra-cluster
    dissemination. Returns (local commit, global commit) per replica."""
    local, reps = {}, []
    for c in range(M):
        mem = [i for i in members if cl[i] == c]
        k = e % len(mem)
        mem = mem[k:] + mem[:k]
        t = t0
        while mem[0] in byz:                      # local view change
            mem = mem[1:] + mem[:1]
            t += 6 * TAU_MIN + 50.0
        r = pbft_round(net, mem, honest, mem[0], t) if len(mem) > 1 else {mem[0]: t}
        local.update(r)
        reps.append((mem[0], mem))
    rep_ids = [r for r, _ in reps]
    t_rep = max(local[r] for r in rep_ids)
    g = pbft_round(net, rep_ids, set(rep_ids), rep_ids[0], t_rep)
    glob = {}
    for r, mem in reps:
        d = net.phase({r: g[r]}, {r: mem}, SIG_B)[r]
        d[r] = g[r]
        glob.update(d)
    return local, glob


def one_run(method, rho, seed, traffic=1000.0):
    rng = np.random.default_rng(seed)               # common random numbers
    cl = np.repeat(np.arange(M), N // M)
    byz_cl = rng.choice(M, F, replace=False)
    byz = {int(c * (N // M) + rng.integers(1, N // M)) for c in byz_cl}
    honest = set(range(N)) - byz
    net = Net(cl, TAU_MIN, rho, np.random.default_rng(seed + 10_000),
              bg_util=bg_util(traffic),
              sig_verify_ms=0.2 if method == "PBFT-Perf" else 1.0)
    maxd = net.base.max()
    orng = np.random.default_rng(seed + 20_000)
    nid = 0
    true = {}
    for c in range(M):
        true[c] = set(range(nid, nid + OBJ_PER_CL)); nid += OBJ_PER_CL
    view = {i: {c: set(true[c]) for c in range(M)} for i in honest}
    gcs = GCS(N, init=np.full(N, 0.95))
    lat, miss, pcc, glat = [], [], [], []
    for e in range(EPOCHS):
        t0 = e * EPOCH
        net.advance(t0)
        for c in range(M):
            k = int(round(TURNOVER * OBJ_PER_CL))
            gone = set(orng.choice(sorted(true[c]), k, replace=False))
            true[c] = (true[c] - gone) | set(range(nid, nid + k)); nid += k
        ref = {c: {o for o in true[c] if orng.random() > MISS} for c in range(M)}
        if method in ("PBFT", "PBFT-Perf"):
            r = flat(net, pbft_round, honest, e % N, t0, 6 * maxd + 50.0)
            arr = {i: {c: r[i] for c in range(M)} for i in honest}
        elif method == "HotStuff":
            r = flat(net, hotstuff_round, honest, e % N, t0, 14 * maxd + 50.0)
            arr = {i: {c: r[i] for c in range(M)} for i in honest}
        else:
            members = (sorted(np.flatnonzero(gcs.admitted()))
                       if method == "AMLC" else list(range(N)))
            local, glob = hier(net, cl, members, honest, byz, e, t0)
            r = local if method == "AMLC" else glob
            glat.append(np.mean([glob[i] - t0 for i in honest if i in glob]))
            arr = {i: {c: (local[i] if (method == "AMLC" and cl[i] == c) else glob[i])
                       for c in range(M)} for i in honest}
        d = np.array([r[i] - t0 for i in honest if i in r])
        if method not in ("AMLC", "ML-PBFT"):
            glat.append(d.mean())
        lat.append(d.mean())
        miss.append(np.mean(d > GAMMA_C))
        tru = set().union(*true.values())
        vals = []
        for i in honest:
            for c in range(M):
                if arr[i][c] - t0 <= GAMMA_C:
                    view[i][c] = set(ref[c])
            o = set().union(*view[i].values())
            vals.append(len(o & tru) / len(o | tru))
        pcc.append(np.mean(vals))
        if method == "AMLC":
            s = [score(0.6, 1.0, 1.0, 1.0) if i in byz else score(1.0, 1.0, 0.98, 1.0)
                 for i in range(N)]
            gcs.update(s)
    return dict(lat=float(np.mean(lat)), dmr=100 * float(np.mean(miss)),
                pcc=100 * float(np.mean(pcc)), glat=float(np.mean(glat)), tx=net.tx_count / EPOCHS)
