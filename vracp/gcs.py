"""GCS-v2 trust score (Eqs. gcs, gcs_v2) with strike rule and probation.

Parameters follow Table III: beta=0.15, initial theta_prob=0.35,
theta_adm=0.55, theta_route=0.60; strike: s<0.3 for 3 consecutive epochs ->
GCS capped at 0.35 for 10 epochs; 3 strikes -> permanent exclusion.
"""
import numpy as np

BETA = 0.15
THETA_PROB = 0.35
THETA_ADM = 0.55
THETA_ROUTE = 0.60
F_HARD = 0.75
STRIKE_S = 0.3
STRIKE_RUN = 3
PROBATION = 10
MAX_STRIKES = 3


def score(F, V, T, A, pom_ok=True):
    """Epoch behaviour score s_i(e), Eq. gcs_v2."""
    d = sorted([F, V, T, A])
    trimmed = 0.5 * (d[1] + d[2])
    hard = 1.0 if F >= F_HARD else 0.0
    return (min(d) ** 2) * np.sqrt(trimmed) * (1.0 if pom_ok else 0.0) * hard


class GCS:
    def __init__(self, n, init=None):
        self.g = np.full(n, THETA_PROB) if init is None else np.array(init, float)
        self.low_run = np.zeros(n, int)
        self.prob_left = np.zeros(n, int)
        self.strikes = np.zeros(n, int)
        self.banned = np.zeros(n, bool)

    def update(self, s, hold=None):
        """Apply one epoch of evidence.

        `hold[i]=True` marks evidence that was not committed within Gamma_c;
        node i's score is then left unchanged (the stale view stays in force),
        which is how committed trust evidence gates relay eligibility.
        """
        s = np.asarray(s, float)
        keep = np.asarray(hold, bool) if hold is not None else np.zeros(len(s), bool)
        g_old, lr_old, pl_old, st_old = (self.g.copy(), self.low_run.copy(),
                                        self.prob_left.copy(), self.strikes.copy())
        self.g = (1 - BETA) * self.g + BETA * s
        low = s < STRIKE_S
        self.low_run = np.where(low, self.low_run + 1, 0)
        hit = self.low_run >= STRIKE_RUN
        self.strikes[hit] += 1
        self.prob_left[hit] = PROBATION
        self.low_run[hit] = 0
        self.banned |= self.strikes >= MAX_STRIKES
        capped = self.prob_left > 0
        self.g[capped] = np.minimum(self.g[capped], THETA_PROB)
        # the strike epoch itself is not counted as a probation epoch
        self.prob_left[capped & ~hit] -= 1
        self.g[self.banned] = 0.0
        if keep.any():                      # roll back uncommitted evidence
            self.g[keep] = g_old[keep]
            self.low_run[keep] = lr_old[keep]
            self.prob_left[keep] = pl_old[keep]
            self.strikes[keep] = st_old[keep]
        return self.g

    def admitted(self):
        return (self.g >= THETA_ADM) & ~self.banned

    def routable(self):
        return (self.g >= THETA_ROUTE) & ~self.banned
