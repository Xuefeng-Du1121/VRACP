"""Measured RDRP break-prediction hit rates, shared by the simulators.

Under SUMO mobility a link break is pre-empted by make-before-break with the
probability that the kinematic predictor (predictor.py) flags it 100-300 ms
before it happens; the rate depends on speed and link kind (v2v, rsu, uav).
The rates are measured once from the traces and cached in predictor.json.
VRACP_P_PRED=<p> forces one value for every speed and link kind.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "predictor.json")
EXP_P_PRED = 0.9
_TABLE = None


def hit(speed_kmh, kind):
    forced = os.environ.get("VRACP_P_PRED")
    if forced:
        return float(forced)
    global _TABLE
    if _TABLE is None:
        if os.path.exists(CACHE):
            _TABLE = json.load(open(CACHE))
        else:
            import predictor
            _TABLE = {str(v): predictor.run(v) for v in predictor.SPEEDS}
            json.dump(_TABLE, open(CACHE, "w"), indent=1)
    key = min(_TABLE, key=lambda k: abs(float(k) - speed_kmh))
    return _TABLE[key][kind]["hit_rate"]
