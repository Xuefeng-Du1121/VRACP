# VRACP simulator: core models (reviewer copy)

This package contains the models and simulators behind the evaluation (Sec. VI) of
*Closed-Loop Trustworthy Coordination for Infrastructure-Deficient Vehicle–Road–Aerial
Mobile Networks* (IEEE TMC submission). The model code is the code that produced the
paper's results; experiment drivers, parameter sweeps, plotting, table generation,
result files and the SUMO traces are not included.

## Modules (`vracp/`)

| File | Content | Paper |
|---|---|---|
| `des.py` | Phase-level message model: intra/cross-cluster delays (`tau_min * rho^u`), 802.11p contention | Sec. III-C, VI-A |
| `consensus.py` | PBFT and HotStuff round models used by every protocol | Sec. IV-A |
| `gcs.py` | GCS-v2 trust score, strike rule, probation, admission and route eligibility | Sec. III-D, Eq. (6) |
| `mobility.py` | Trace-driven link lifetimes (V2V, vehicle–RSU, vehicle–UAV) from SUMO traces | Sec. VI-A |
| `sumo_gen.py` | SUMO scenario generator (20-km two-way road, 1200 veh/h/dir, 5 traces per speed) | Sec. VI-A |
| `predictor.py` | RDRP kinematic link-break predictor (Algorithm 2) and its hit / false-alarm measurement | Sec. IV-B, VI-C |
| `predictor_rates.py` | Measured hit rates fed to the simulators (computed from the traces on first use) | Sec. VI-C |
| `consensus_sim.py` | Consensus under delay heterogeneity: PBFT, ML-PBFT, PBFT-Perf, HotStuff, AMLC | Sec. VI-B |
| `routing_sim.py` | Routing latency vs. hop count: AODV-U, SZLS-GPSR, QLR, Trust-AODV, RDRP | Sec. VI-C |
| `sysim.py` | System simulator of the closed loop (AMLC, trust, RDRP, ITAC, three interfaces), all ablation and baseline variants, badmouthing | Sec. VI-C–VI-H |
| `itac_learner.py` | Asynchronous / barrier-synchronized actor–critic with feasibility projection in place of ITAC's rule | Sec. VI-D, Algorithm 3 |
| `semantic_sim.py` | Semantic-message workload (TPR, COR, Stale, FOR) under Mild / Default / Harsh stress | Sec. VI-F |

Each file's docstring lists its model assumptions. Configuration names follow the
paper: `sysim.VARIANTS` holds `Full VRACP`, `w/o AMLC`, `w/o Trust`, `w/o Predict`,
`w/o Async`, `w/o A->R`, `w/o R->I`, `w/o I->A`, `w/o Feedback (all)`, `Stchd.-Naive`,
`Stchd.-Cons`, `Stchd.-Two` and `HierHotStuff`.

## Requirements

Python 3.9+ and NumPy. Generating the SUMO traces additionally needs pandas and SUMO
(`pip install eclipse-sumo`; set `SUMO_BIN` if `sumo` is not on `PATH`).

## Usage

Exponential link lifetimes (no traces needed):

```python
import sysim, routing_sim, consensus_sim, semantic_sim
r = sysim.Sim("Full VRACP", N=24, f=4, speed_kmh=20.0, B_kb=1000.0, snr_db=15.0, seed=5000).run()
print(r["PCC"], r["PDR"], r["DSR"], r["tau"])
print(routing_sim.one_run("VRACP", 6, 3000)["lat"])          # six-hop latency, ms
print(consensus_sim.one_run("AMLC", 40, 6000))                 # rho_delay = 40
print(semantic_sim.one_run("VRACP", "Harsh", 4000))
```

Trace-driven mobility, as in the paper:

```bash
cd vracp && python3 sumo_gen.py 20 60 120        # writes sumo/v<speed>/s<k>/trace.npz
```

```python
r = sysim.Sim("Full VRACP", 24, 4, 120.0, 1000.0, 15.0, 5000, mobility="sumo").run()
routing_sim.MOBILITY, routing_sim.SPEED = "sumo", 120.0
```

The first SUMO run measures the predictor's hit rates on the traces and caches them in
`vracp/predictor.json`. Every value in the paper is the mean over 10 seeds with common
random numbers across configurations.
