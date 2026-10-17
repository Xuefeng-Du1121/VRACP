"""Consensus round models: PBFT, ML-PBFT, PBFT-Perf, HotStuff, AMLC (Sec. VI-B).

One run = E epochs x R rounds. Each round commits one reference digest.
Byzantine replicas are silent (no votes) and, when not excluded, inject
forged objects into their local reports. A Byzantine leader stalls a round
until the view-change timeout. AMLC additionally runs GCS admission (Eq. gcs)
at epoch boundaries and reports the local (intra-cluster) commit latency.

PCC (Eq. PCC) is computed with a semantic object model: the true object set
turns over between rounds; a node whose reference was not committed within the
round deadline keeps its last committed (stale) set.
"""
import numpy as np
from des import Net
from gcs import GCS

SIG_B = 256          # vote / certificate size (bytes)
PROP_B = 1500        # proposal carrying a digest batch (bytes)


def f_of(n):
    return (n - 1) // 3


def qsize(n):
    """Byzantine quorum for n replicas tolerating f=floor((n-1)/3)."""
    return int(np.ceil((n + f_of(n) + 1) / 2))


def quorum_time(times, q):
    """Time at which q messages have arrived (inf if fewer than q)."""
    t = sorted(times)
    return t[q - 1] if len(t) >= q else np.inf


# --------------------------------------------------------------------- PBFT
def pbft_round(net, members, honest, leader, t0, verify_ms=None, batch=1):
    """Three-phase PBFT among `members`; returns commit time per honest node."""
    n = len(members)
    q = qsize(n)
    if leader not in honest:
        return None
    # pre-prepare
    pp = net.phase({leader: t0}, {leader: members}, PROP_B * batch, verify_ms)[leader]
    pp[leader] = t0
    # prepare: honest replicas broadcast after receiving the pre-prepare
    senders = {i: pp[i] for i in members if i in honest and i in pp}
    pr = net.phase(senders, {i: members for i in senders}, SIG_B, verify_ms)
    prepared = {}
    for i in members:
        if i not in honest or i not in pp:
            continue
        arr = [pr[s][i] for s in pr if i in pr[s]] + [pp[i]]
        prepared[i] = max(pp[i], quorum_time(arr, q))
    senders = {i: t for i, t in prepared.items() if np.isfinite(t)}
    if not senders:
        return {}
    cm = net.phase(senders, {i: members for i in senders}, SIG_B, verify_ms)
    commit = {}
    for i in senders:
        arr = [cm[s][i] for s in cm if i in cm[s]] + [senders[i]]
        commit[i] = max(senders[i], quorum_time(arr, q))
    return commit


# ----------------------------------------------------------------- HotStuff
def hotstuff_round(net, members, honest, leader, t0):
    """Basic HotStuff: 4 leader broadcasts and 3 vote collections (7 one-way
    delays), threshold-signature certificates (linear messages)."""
    n = len(members)
    q = qsize(n)
    if leader not in honest:
        return None
    t = t0
    for ph in range(3):
        bc = net.phase({leader: t}, {leader: members}, PROP_B if ph == 0 else SIG_B)[leader]
        voters = {i: bc[i] for i in members if i in honest and i in bc and i != leader}
        votes = net.phase(voters, {i: [leader] for i in voters}, SIG_B)
        arr = [votes[s][leader] for s in votes] + [t]
        t = quorum_time(arr, q)
        if not np.isfinite(t):
            return {}
    dec = net.phase({leader: t}, {leader: members}, SIG_B)[leader]
    dec[leader] = t
    return {i: dec[i] for i in members if i in honest and i in dec}


