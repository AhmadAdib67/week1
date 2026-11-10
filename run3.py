"""Monday 3: proximity vs proximity+motion detectors -> same CA -> export detection + simulation outputs,
with controls that separate detector mistakes from simulator-rule effects."""
import os, itertools, numpy as np, pandas as pd
import data, detect, evaluate as ev, ca
os.makedirs("outputs3", exist_ok=True)
rng0 = np.random.default_rng(123)
df, _ = data.load_eth("data"); dev, test = ev.time_split(df)

# ---- 1. tune both detectors on DEV only (same grid size philosophy; v_min fixed at 0.3 m/s)
rows = []
for eps in [0.5, 0.75, 1.0, 1.25, 1.5]:
    rows.append(dict(method="proximity", eps=eps, cos_min=np.nan, **ev.prf(ev.frame_pairs(dev, detect.detect_proximity(dev, eps)))))
    for c in [0.5, 0.8, 0.9]:
        rows.append(dict(method="motion", eps=eps, cos_min=c, **ev.prf(ev.frame_pairs(dev, detect.detect_motion(dev, eps, c)))))
sw = pd.DataFrame(rows); sw.to_csv("outputs3/dev_sweep.csv", index=False)
bp = sw[sw.method == "proximity"].sort_values("f1").iloc[-1]; bm = sw[sw.method == "motion"].sort_values("f1").iloc[-1]
print("best DEV proximity:", dict(eps=bp.eps, f1=round(bp.f1, 3)), "| motion:", dict(eps=bm.eps, cos=bm.cos_min, f1=round(bm.f1, 3)))
P = lambda d: detect.detect_proximity(d, bp.eps, config_id="test")
M = lambda d: detect.detect_motion(d, bm.eps, bm.cos_min, config_id="test")

# ---- 2. TEST evaluation on identical eligible people/frames; export predictions + metrics
pp, pm = P(test), M(test)
pd.concat([pp, pm]).to_csv("outputs3/predictions_test.csv", index=False)
res = []
for name, pr in [("proximity", pp), ("motion", pm)]:
    pr_ = ev.frame_pairs(test, pr); res.append(dict(method=name, **ev.prf(pr_)))
R = pd.DataFrame(res); R.to_csv("outputs3/detection_metrics.csv", index=False); print("\nTEST detection\n", R.round(3).to_string(index=False))

# ---- 3. CA start frames: TEST block only, most reference-group members, >=100 frames apart
cnt = test[test.ref_group_id >= 0].groupby("frame").size().sort_values(ascending=False)
frames = []
for f, c in cnt.items():
    if c >= 4 and all(abs(f - g) >= 100 for g in frames): frames.append(int(f))
    if len(frames) == 8: break
wall, ex = ca.parse(); SEEDS = range(20)

def part(pred, pids):   # prediction -> {pid: group or None}
    s = pred.set_index("person_id").pred_group_id
    return {p: (int(s[p]) if s[p] >= 0 else None) for p in pids}

runs, logs, per_frame = [], [], []
for f in frames:
    g0 = test[test.frame == f]; pos0 = ca.place_from_tracks(df, f, wall); pids = list(pos0)
    ref = {p: (int(r) if r >= 0 else None) for p, r in zip(g0.person_id, g0.ref_group_id)}
    prox, mot = part(pp[pp.frame == f], pids), part(pm[pm.frame == f], pids)
    # CONTROL: shuffle the inferred-motion membership among people (same group-size structure, wrong people)
    vals = list(mot.values()); rng0.shuffle(vals); shuf = dict(zip(pids, vals))
    scen = {"individual": ({p: None for p in pids}, False), "reference": (ref, True),
            "inferred_proximity": (prox, True), "inferred_motion": (mot, True), "shuffled_control": (shuf, True)}
    # detector error at THIS frame (simulator-independent): pair counts on the people placed in the CA
    fp_ = lambda pr: ev.prf(ev.frame_pairs(g0, pr[pr.frame == f]))
    per_frame.append(dict(frame=f, n_ped=len(pids), n_ref_grouped=sum(v is not None for v in ref.values()),
                          **{f"prox_{k}": v for k, v in fp_(pp).items() if k in ("TP", "FP", "FN")},
                          **{f"mot_{k}": v for k, v in fp_(pm).items() if k in ("TP", "FP", "FN")}))
    for name, (grp, coh) in scen.items():
        for sd in SEEDS:   # SAME seed -> same walking speeds in every scenario (common random numbers)
            st = {}
            log, es = ca.simulate(pos0, grp, wall, ex, seed=sd, cohesion=coh, stats=st, ref=ref)
            runs.append(dict(start_frame=f, scenario=name, seed=sd, waits=st.get("waits", 0), false_waits=st.get("false_waits", 0),
                             **ca.metrics(es, ref, len(pids))))   # ALWAYS scored against reference groups
            if sd == 0:
                log = log.assign(scenario=name, seed=sd, start_frame=f); logs.append(log)
runs = pd.DataFrame(runs); runs.to_csv("outputs3/ca_runs.csv", index=False)
pd.concat(logs).to_csv("outputs3/ca_trajectories_seed0.csv", index=False)
PF = pd.DataFrame(per_frame)

# ---- 4. decomposition: paired differences (same frame + seed), averaged within frame, then bootstrap over frames
piv = runs.pivot_table(index=["start_frame", "seed"], columns="scenario", values=["group_exit_gap", "evac_time", "mean_exit", "false_waits"])
def contrast(metric, a, b):
    d = (piv[metric][a] - piv[metric][b]).groupby("start_frame").mean().dropna().values
    bs = [rng0.choice(d, len(d)).mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)
C = []
for metric in ["group_exit_gap", "evac_time", "mean_exit"]:
    for label, a, b in [("SIMULATOR RULE effect (reference - individual)", "reference", "individual"),
                        ("DETECTOR cost, proximity (inferred_prox - reference)", "inferred_proximity", "reference"),
                        ("DETECTOR cost, motion (inferred_motion - reference)", "inferred_motion", "reference"),
                        ("CONTROL: random groups (shuffled - reference)", "shuffled_control", "reference"),
                        ("motion vs proximity (motion - proximity)", "inferred_motion", "inferred_proximity")]:
        m, lo, hi, n = contrast(metric, a, b); C.append(dict(metric=metric, contrast=label, mean_diff=m, ci95_lo=lo, ci95_hi=hi, n_frames=n))
C = pd.DataFrame(C); C.to_csv("outputs3/decomposition.csv", index=False)
W = runs.groupby("scenario")[["waits", "false_waits"]].mean().round(2)
print("\nPer-frame detector errors (CA population):\n", PF.to_string(index=False))
print("\nCohesion waits per run (false_waits = wait triggered by a non-reference group):\n", W.to_string())
print("\nDecomposition (paired, bootstrap over", len(frames), "start frames):\n", C.round(2).to_string(index=False))
PF.to_csv("outputs3/ca_frame_detector_errors.csv", index=False)
print("\nexported -> outputs3/")
