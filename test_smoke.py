"""Synthetic smoke tests (NOT real-data evaluation). Run: python test_smoke.py"""
import numpy as np, pandas as pd, data, detect, evaluate as ev, ca
# schema test
try: data.validate(pd.DataFrame({"frame": [1], "x": [0.]})); raise SystemExit("should fail")
except ValueError: pass
# toy hand-checked pairwise counts: A,B walk together (ref group 0); C passes 0.5 m from B
toy = pd.DataFrame({"frame": [1]*3, "person_id": [1, 2, 3], "x": [0, .5, 1.0], "y": [0, 0, 0], "ref_group_id": [0, 0, -1]})
p = detect.detect_proximity(toy, eps=0.6)     # chain 1-2-3 -> one cluster: pairs (1,2),(1,3),(2,3) predicted
m = ev.prf(ev.frame_pairs(toy, p)); assert (m["TP"], m["FP"], m["FN"]) == (1, 2, 0), m   # 1 true pair, 2 false merges
p = detect.detect_proximity(toy, eps=0.4); m = ev.prf(ev.frame_pairs(toy, p)); assert (m["TP"], m["FP"], m["FN"]) == (0, 0, 1), m
# CA: no wall crossing, one ped per cell, everyone exits
wall, ex = ca.parse(); pos = {i: (2+i, 2) for i in range(6)}
log, es = ca.simulate(pos, {i: i//2 for i in range(6)}, wall, ex, seed=1)
assert len(es) == 6 and not wall[log.cell_y, log.cell_x].any()
assert not log.duplicated(["step", "cell_x", "cell_y"]).any()
# regression: cohesion must never deadlock the exit
for sd in range(40):
    _, es = ca.simulate({i: (2+i, 1+i % 3) for i in range(8)}, {i: i//2 for i in range(8)}, wall, ex, seed=sd)
    assert len(es) == 8, sd
print("all smoke tests passed")
