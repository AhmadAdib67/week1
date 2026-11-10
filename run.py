"""End-to-end: import+replay -> DBSCAN baseline -> evaluate -> failure analysis -> CA scenarios."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data, detect, evaluate as ev, ca
os.makedirs("outputs", exist_ok=True)

# 1. import + validate + replay
df, groups = data.load_eth("data")
rep = data.validate(df, expect_ids=390, expect_rows=6544); print("VALIDATION", rep)
print("replay ->", data.replay(df, "data", "outputs/replay.gif"))

# 2-3. DBSCAN baseline; eps tuned on DEV block only, reported on TEST block
dev, test = ev.time_split(df)
rows = []
for eps in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    rows.append(dict(eps=eps, **ev.prf(ev.frame_pairs(dev, detect.detect_proximity(dev, eps)))))
sweep = pd.DataFrame(rows); print("\nDEV sweep\n", sweep.round(3).to_string(index=False))
best = float(sweep.loc[sweep.f1.idxmax(), "eps"])
pt = detect.detect_proximity(test, best, config_id="test"); pairs_t = ev.frame_pairs(test, pt)
res = ev.prf(pairs_t); print(f"\nTEST (eps={best}):", {k: round(v, 3) for k, v in res.items()})
pt.to_csv("outputs/predictions_test.csv", index=False)
pd.DataFrame([dict(eps=best, split="test", **res)]).to_csv("outputs/metrics.csv", index=False)
sweep.to_csv("outputs/dev_sweep.csv", index=False)

# 5. incorrect detection: worst false merge (FP pair) + worst miss (FN pair), with diagnostics
def diag(a, b, d):
    A, B = d[d.person_id == a].set_index("frame"), d[d.person_id == b].set_index("frame")
    c = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    dist = np.hypot(c.x_a-c.x_b, c.y_a-c.y_b)
    va, vb = c[["vx_a", "vy_a"]].values, c[["vx_b", "vy_b"]].values
    cos = (va*vb).sum(1)/(np.linalg.norm(va, axis=1)*np.linalg.norm(vb, axis=1)+1e-9)
    return c, dict(shared_frames=len(c), mean_dist=round(dist.mean(), 2), mean_cos_heading=round(cos.mean(), 2),
                   speed_a=round(np.linalg.norm(va, axis=1).mean(), 2), speed_b=round(np.linalg.norm(vb, axis=1).mean(), 2))
fp = pairs_t[~pairs_t.ref & pairs_t.pred].groupby(["a", "b"]).size().sort_values(ascending=False)
fn = pairs_t[pairs_t.ref & ~pairs_t.pred].groupby(["a", "b"]).size().sort_values(ascending=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (title, pr) in zip(axs, [("FALSE MERGE", fp), ("FALSE SPLIT", fn)]):
    if pr.empty: continue
    (a, b), n = pr.index[0], pr.iloc[0]; c, dd = diag(a, b, test); print(f"\n{title}: pair ({a},{b}), wrong in {n} frames", dd)
    for who, col in [("a", "C0"), ("b", "C3")]:
        ax.plot(c[f"x_{who}"], c[f"y_{who}"], "o-", color=col, ms=3, label=f"ped {a if who=='a' else b}")
        ax.annotate("start", (c[f"x_{who}"].iloc[0], c[f"y_{who}"].iloc[0]), color=col)
    ax.set_title(f"{title}: ped {a} & {b}, wrong in {n} frames\nmean dist {dd['mean_dist']} m, heading cos {dd['mean_cos_heading']}", fontsize=9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal"); ax.legend()
plt.tight_layout(); plt.savefig("outputs/failure_cases.png", dpi=120); plt.close()

# 4. CA with reference group membership (+ individual and inferred-from-DBSCAN for comparison)
wall, ex = ca.parse()
cnt = df[df.ref_group_id >= 0].groupby("frame").size(); f0 = int(cnt.idxmax())
g0 = df[df.frame == f0]; pos0 = ca.place_from_tracks(df, f0, wall)
pred0 = detect.detect_proximity(g0, best).set_index("person_id").pred_group_id
scen = {"individual": {p: None for p in pos0},
        "reference": {p: (int(r) if r >= 0 else None) for p, r in zip(g0.person_id, g0.ref_group_id)},
        "inferred": {p: (int(l) if l >= 0 else None) for p, l in pred0.items()}}
print(f"\nCA start frame {f0}: {len(pos0)} peds, {sum(v is not None for v in scen['reference'].values())} in reference groups")
allm, logs = [], []
for name, grp in scen.items():
    for seed in range(30):
        log, es = ca.simulate(pos0, grp, wall, ex, seed=seed, cohesion=(name != "individual"))
        allm.append(dict(scenario=name, seed=seed, **ca.metrics(es, scen["reference"], len(pos0))))  # always score vs REFERENCE groups
        if seed == 0: log["scenario"] = name; log["seed"] = 0; logs.append(log)
M = pd.DataFrame(allm); S = M.groupby("scenario")[["evac_time", "mean_exit", "exited", "group_exit_gap"]].agg(["mean", "std"]).round(2)
print(S.to_string()); M.to_csv("outputs/ca_metrics.csv", index=False)
pd.concat(logs).to_csv("outputs/ca_trajectories_seed0.csv", index=False)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, t in zip(axs, [0, 12, 25]):
    L = logs[1][logs[1].step == t]; ax.imshow(~wall, cmap="gray", vmin=-1, vmax=1)
    ax.scatter(*ex, marker="s", c="g", s=90)
    ax.scatter(L.cell_x, L.cell_y, c=[(-1 if g != g or g is None else g) for g in L.group_id], cmap="tab10", s=70, edgecolors="k")
    ax.set_title(f"CA 'reference' scenario, step {t}", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig("outputs/ca_snapshots.png", dpi=120); plt.close()
print("\nSaved to outputs/")
