"""Generate SUMO mobility traces for the corridor scenario of sysim.py.

Usage: python3 sumo_gen.py [speed_kmh ...]      (default: 20 60 120)
       K independent traces per speed (SUMO seed + initial placement).

Scenario (declared a priori, not tuned to results)
  * Straight two-way road, L = 20 km, 2 lanes per direction (3.2-m lanes),
    speed limit = the nominal speed of the experiment. RSUs every 1 km on the
    roadside; experiments bind their vehicles to the middle 6 km
    [7, 13] km, so the 6-RSU corridor of sysim.py sees no boundary effects.
  * Demand: 1200 veh/h per direction (600 veh/h/lane), the same flow at all
    speeds, so density falls as speed rises (60 -> 10 veh/km/direction).
    The road is pre-filled at that density (exponential headways, >= 10 m) and
    replenished by inflow at both ends.
  * Vehicles: Krauss car following (sigma = 0.5), LC2013 lane changing,
    speed factor ~ N(1, 0.1) truncated to [0.8, 1.2] (SUMO's default spread).
  * 300 s warm-up, then 600 s of FCD at 0.5-s steps (--eager-insert).
Output: sumo/v<speed>/s<k>/trace.npz with x, y [n_veh, n_step] (NaN = not on road),
dt and t0.
"""
import os, shutil, sys, subprocess
import numpy as np
import pandas as pd

L = 20000.0
LANES = 2
FLOW = 1200.0          # veh/h per direction
WARM = 300.0
REC = 600.0
STEP = 0.5
K = 5                  # traces per speed
# SUMO binaries: $SUMO_BIN, else the directory of `sumo` on PATH (pip install eclipse-sumo)
BIN = os.environ.get("SUMO_BIN") or os.path.dirname(shutil.which("sumo") or "") or os.path.expanduser("~/Library/Python/3.9/bin")


def sh(*a):
    subprocess.run(a, check=True, stdout=subprocess.DEVNULL)


def build(speed_kmh, out, k=0):
    os.makedirs(out, exist_ok=True)
    v = speed_kmh / 3.6
    p = lambda f: os.path.join(out, f)
    open(p("n.nod.xml"), "w").write(
        f'<nodes><node id="W" x="0" y="0"/><node id="E" x="{L:.0f}" y="0"/></nodes>\n')
    open(p("n.edg.xml"), "w").write(
        '<edges>'
        f'<edge id="WE" from="W" to="E" numLanes="{LANES}" speed="{v:.3f}"/>'
        f'<edge id="EW" from="E" to="W" numLanes="{LANES}" speed="{v:.3f}"/>'
        '</edges>\n')
    sh(f"{BIN}/netconvert", "-n", p("n.nod.xml"), "-e", p("n.edg.xml"),
       "-o", p("net.xml"), "--no-turnarounds", "true")
    rng = np.random.default_rng(int(speed_kmh) * 100 + k)
    dens = FLOW / speed_kmh / LANES / 1000.0           # veh per m per lane
    rows = []
    for e in ["WE", "EW"]:
        for ln in range(LANES):
            x = rng.exponential(1 / dens)
            while x < L - 10:
                rows.append((e, ln, x))
                x += max(10.0, rng.exponential(1 / dens))
    with open(p("r.rou.xml"), "w") as f:
        f.write('<routes>\n'
                '<vType id="car" length="4.5" minGap="2.5" sigma="0.5" '
                'speedFactor="normc(1,0.1,0.8,1.2)" carFollowModel="Krauss" '
                'laneChangeModel="LC2013"/>\n'
                '<route id="rWE" edges="WE"/><route id="rEW" edges="EW"/>\n')
        for i, (e, ln, x) in enumerate(rows):
            f.write(f'<vehicle id="i{i}" type="car" route="r{e}" depart="0" '
                    f'departLane="{ln}" departPos="{x:.1f}" departSpeed="desired"/>\n')
        end = WARM + REC + 10
        for e in ["WE", "EW"]:
            f.write(f'<flow id="f{e}" type="car" route="r{e}" begin="0" end="{end:.0f}" '
                    f'vehsPerHour="{FLOW:.0f}" departLane="best" departSpeed="desired"/>\n')
        f.write('</routes>\n')
    sh(f"{BIN}/sumo", "-n", p("net.xml"), "-r", p("r.rou.xml"),
       "--step-length", str(STEP), "--begin", "0", "--end", str(WARM + REC),
       "--fcd-output", p("fcd.csv"), "--device.fcd.begin", str(WARM),
       "--seed", str(int(speed_kmh) * 100 + k), "--no-step-log", "true",
       "--no-warnings", "true", "--collision.action", "none", "--eager-insert", "true")
    df = pd.read_csv(p("fcd.csv"), sep=";", usecols=["timestep_time", "vehicle_id",
                                                     "vehicle_x", "vehicle_y"])
    df = df.dropna()
    ids = {k: i for i, k in enumerate(df["vehicle_id"].unique())}
    ks = np.rint((df["timestep_time"].values - WARM) / STEP).astype(int)
    n_t = int(round(REC / STEP)) + 1
    X = np.full((len(ids), n_t), np.nan, np.float32)
    Y = np.full((len(ids), n_t), np.nan, np.float32)
    vi = df["vehicle_id"].map(ids).values
    X[vi, ks] = df["vehicle_x"].values
    Y[vi, ks] = df["vehicle_y"].values
    np.savez_compressed(p("trace.npz"), x=X, y=Y, dt=STEP, t0=WARM, speed=speed_kmh)
    os.remove(p("fcd.csv"))
    on = ~np.isnan(X)
    mid = on & (X > 7000) & (X < 13000)
    print(f"{speed_kmh:g} km/h s{k}: {len(ids)} vehicles, {on.sum(0).mean():.0f} on road, "
          f"{mid.sum(0).mean() / 12:.1f} veh/km/dir in [7,13] km")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for s in (sys.argv[1:] or ["20", "60", "120"]):
        for k in range(K):
            build(float(s), os.path.join(here, "sumo", f"v{int(float(s))}", f"s{k}"), k)
