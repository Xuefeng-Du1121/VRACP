"""Routing simulator: end-to-end latency vs. minimum hop count (Sec. VI-C).

One CBR flow (10 Hz, 1000-B packets) over a route of H minimum hops for
60 s; latency = delivery time - first generation time of the packet
(including end-to-end retransmissions). Parameters are fixed a priori:

  * Per-hop sojourn = (#MAC attempts) * 2.27 ms + Exp(W) queueing, with
    frame error rate 10%, retry limit 7, background utilisation u = 0.5
    (M/D/1 mean wait W = u/(2(1-u)) * 2.27 ms).
  * Link lifetime ~ Exp(10 s). Link-stability-aware selection
    (SZLS-GPSR, VRACP) picks the more stable of two candidates (max of two
    draws). With MOBILITY = "sumo" each hop's lifetime is the residual
    lifetime of a real endpoint pair from SUMO traces (mobility.py): hop 0 is
    the RSU hop, the others V2V; hop h is anchored at its own bound vehicle,
    and the stability-aware methods take the longer of two candidate links.
    The exponential draws are still made and discarded so that all other
    draws stay aligned with the exp runs.
  * Relays are Byzantine with probability f/N = 12.5% and drop 50% of
    forwarded packets; a lost packet is retransmitted by the source after
    RTO = 200 ms (at most 3 retransmissions).
  * Link break handling:
      AODV-U / Trust-AODV: link-layer detection after the retry limit; the
        packet is dropped (no salvaging); RERR travels back to the source,
        then RREQ flood (10 ms jitter per hop) + RREP over H hops; packets
        generated meanwhile wait at the source.
      SZLS-GPSR: local greedy re-selection after the retry limit; with
        probability 0.1 a local void forces perimeter mode (+2 hops).
      QLR: local re-selection after the retry limit (next-best Q value).
      VRACP: RDRP predicts the break (probability 0.9 under the exponential
        model, the measured hit rate of predictor_rates.py under SUMO) and
        switches to a prepared backup link without interruption; otherwise
        local backup switch after the retry limit.
  * Byzantine relays: Trust-AODV (forwarding-ratio trust) and VRACP (GCS
    route eligibility) exclude them at route (re)selection; QLR replaces a
    relay after observing 3 losses through it; AODV-U and SZLS-GPSR do not
    filter.
"""
import numpy as np
import predictor_rates as PR

SLOT_T = 2.27
P_LOSS = 0.10
RETRY = 7
U = 0.5
W = U / (2 * (1 - U)) * SLOT_T
LIFE = 10000.0
P_BYZ = 0.125
BYZ_DROP = 0.5
RTO = 200.0
MAX_RTX = 3
JITTER = 10.0
P_VOID = 0.1
QLR_LOSS = 3          # QLR replaces a relay after this many observed losses
P_PRED = 0.9
RATE = 100.0
DUR = 60000.0
METHODS = ["SZLS-GPSR", "AODV-U", "Trust-AODV", "QLR", "VRACP"]
STABLE = {"SZLS-GPSR", "VRACP"}
MOBILITY, SPEED = "exp", 20.0      # "exp" or "sumo"; vehicle speed (km/h)
TRUST = {"Trust-AODV", "VRACP"}
REACTIVE = {"AODV-U", "Trust-AODV"}


def one_run(method, H, seed):
    rng = np.random.default_rng(seed)
    g = np.random.default_rng(seed + 1)          # common across methods
    mob = None
    if MOBILITY == "sumo":
        from mobility import Mobility
        mob = Mobility(SPEED, seed + 5, H)

    def life(h, t):
        x = rng.exponential(LIFE)
        x = max(x, rng.exponential(LIFE)) if method in STABLE else x
        if mob is not None:
            kind = "rsu" if h == 0 else "v2v"
            x = mob.life(kind, h, t)
            if method in STABLE:
                x = max(x, mob.life(kind, h, t))
        return x

    def p_pred(h):
        # hop 0 is the vehicle--RSU hop, the others V2V (as in life())
        if mob is None:
            return P_PRED
        return PR.hit(SPEED, "rsu" if h == 0 else "v2v")

    def relay():
        byz = rng.random() < P_BYZ
        if method in TRUST:
            byz = False
        return dict(byz=byz, loss=0)

    def hop_time():
        att = 1
        while rng.random() < P_LOSS and att < RETRY:
            att += 1
        return att * SLOT_T + rng.exponential(W)

    def new_route(t):
        return dict(brk=[t + life(h, t) for h in range(H)],
                    pred=[rng.random() < p_pred(h) for h in range(H)],
                    rel=[relay() for _ in range(H - 1)], extra=0)

    route = new_route(0.0)
    up = 0.0
    lat = []
    for t_gen in np.arange(g.uniform(0, RATE), DUR, RATE):
        t_first = t_gen
        t = t_gen
        for _ in range(MAX_RTX + 1):
            t = max(t, up)
            t_send = t
            ok = True
            k = 0
            hops = H + route["extra"]
            while k < hops:
                kk = min(k, H - 1)
                if route["brk"][kk] <= t:
                    if method == "VRACP" and route["pred"][kk]:
                        pass                                  # pre-switched
                    elif method in REACTIVE:
                        t += RETRY * SLOT_T
                        back = sum(hop_time() for _ in range(k))
                        disc = sum(hop_time() + rng.uniform(0, JITTER) for _ in range(H)) \
                            + sum(hop_time() for _ in range(H))
                        up = t + back + disc
                        route = new_route(up)
                        ok = False
                        break
                    else:
                        t += RETRY * SLOT_T
                        if method == "SZLS-GPSR" and rng.random() < P_VOID:
                            route["extra"] += 2
                            hops += 2
                    route["brk"][kk] = t + life(kk, t)
                    route["pred"][kk] = rng.random() < p_pred(kk)
                t += hop_time()
                if 0 < k + 1 < H:                              # relay k+1 forwards
                    r = route["rel"][k]
                    if r["byz"] and rng.random() < BYZ_DROP:
                        r["loss"] += 1
                        if method == "QLR" and r["loss"] >= QLR_LOSS:
                            route["rel"][k] = dict(byz=rng.random() < P_BYZ, loss=0)
                        ok = False
                        break
                k += 1
            if ok:
                lat.append(t - t_first)
                break
            # reactive repair: retransmit once the new route is up;
            # silent drop: source retransmits after RTO
            t = up if up > t_send else t_send + RTO
        route["extra"] = 0 if rng.random() < 0.5 else route["extra"]   # perimeter exit
    return dict(lat=float(np.mean(lat)), pdr=100 * len(lat) / len(np.arange(0, DUR, RATE)))
