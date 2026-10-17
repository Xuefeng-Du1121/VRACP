"""Phase-level message model shared by all consensus protocols.

Topology: nodes are grouped into geographic clusters; each cluster has one
shared 802.11p channel (FIFO server). A message transmitted by a node occupies
its cluster channel once (broadcast medium). Copies for other clusters are
carried over the RSU/UAV backhaul with the cluster-pair one-way delay and then
re-transmitted once on the destination cluster channel. Every receiver verifies
each message on its own FIFO CPU.

Within one protocol phase all transmissions are served in time order on each
channel; channel and CPU occupancy carry over between phases.
"""
import numpy as np


class Net:
    def __init__(self, cluster_of, tau_min, rho, rng, phy_mbps=6.0,
                 bg_util=0.07, sig_verify_ms=1.0, jitter=0.25):
        self.cl = np.asarray(cluster_of)
        self.n = len(self.cl)
        self.m = int(self.cl.max()) + 1
        self.rng = rng
        self.phy = phy_mbps
        self.bg = bg_util
        self.sig = sig_verify_ms
        self.jit = jitter
        # Base one-way delay: intra-cluster tau_min; cluster pair (a,b):
        # tau_min * rho**u, u ~ U(0.5,1), the largest pair pinned to rho so
        # that rho_delay = max/min holds for the base delays.
        base = np.full((self.m, self.m), float(tau_min))
        pairs = [(a, b) for a in range(self.m) for b in range(a + 1, self.m)]
        if rho > 1 and pairs:
            u = rng.uniform(0.5, 1.0, len(pairs))
            u[int(np.argmax(u))] = 1.0
            for (a, b), ui in zip(pairs, u):
                base[a, b] = base[b, a] = tau_min * rho ** ui
        self.base = base
        self.ch_busy = [[] for _ in range(self.m)]   # sorted busy intervals
        self.cpu_free = np.zeros(self.n)
        self.tx_count = 0
        self.bytes = 0

    def _occupy(self, c, t, dur):
        """First-fit a transmission of length dur on channel c at or after t."""
        busy = self.ch_busy[c]
        # drop intervals that ended long ago
        while busy and busy[0][1] < t - 5000.0:
            busy.pop(0)
        st = t
        k = 0
        for k, (a, b) in enumerate(busy):
            if b <= st:
                continue
            if a >= st + dur:
                break
            st = b
        else:
            k = len(busy)
        import bisect
        bisect.insort(busy, (st, st + dur))
        return st + dur

    def advance(self, t):
        """Start of a new epoch: resources are free no earlier than t."""
        self.cpu_free[:] = np.maximum(self.cpu_free, t)

    def tx_time(self, size_b):
        # 0.2 ms AIFS/backoff/preamble + serialization, inflated by the
        # background data-plane utilization of the channel.
        return (0.2 + size_b * 8.0 / (self.phy * 1e3)) / (1.0 - self.bg)

    def _delay(self, a, b):
        return self.base[a, b] * self.rng.lognormal(0.0, self.jit)

    def phase(self, send_times, dests, size_b, verify_ms=None):
        """send_times: {src: t}; dests: {src: iterable of receivers}.

        Returns {src: {dst: time the verified message is available at dst}}.
        """
        v = self.sig if verify_ms is None else verify_ms
        tx = self.tx_time(size_b)
        # stage 1: source-channel transmissions in time order
        done = {}
        for s in sorted(send_times, key=send_times.get):
            done[s] = self._occupy(self.cl[s], send_times[s], tx)
            self.tx_count += 1
            self.bytes += size_b
        # stage 2: relay jobs on destination channels
        relay = {}
        arrive = {}
        for s, t_done in done.items():
            cs = self.cl[s]
            arrive[s] = {}
            for c in {self.cl[d] for d in dests[s] if d != s}:
                if c == cs:
                    continue
                relay.setdefault(c, []).append((t_done + self._delay(cs, c), s))
        relay_done = {}
        for c, jobs in relay.items():
            jobs.sort()
            for t, s in jobs:
                relay_done[(s, c)] = self._occupy(c, t, tx)
                self.tx_count += 1
                self.bytes += size_b
        # stage 3: per-receiver verification in arrival order
        per_rx = {}
        for s, t_done in done.items():
            cs = self.cl[s]
            for d in dests[s]:
                if d == s:
                    continue
                cd = self.cl[d]
                t = t_done + self._delay(cs, cs) if cd == cs else relay_done[(s, cd)]
                per_rx.setdefault(d, []).append((t, s))
        for d, items in per_rx.items():
            items.sort()
            for t, s in items:
                st = max(t, self.cpu_free[d])
                self.cpu_free[d] = st + v
                arrive[s][d] = st + v
        return arrive
