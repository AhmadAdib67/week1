"""Pairwise co-membership evaluation (precision / recall / F1) on the same eligible frames."""
import numpy as np, pandas as pd
from itertools import combinations
from data import UNLABELLED

def frame_pairs(df, pred):
    """One row per (frame, person pair) with ref/pred co-membership flags."""
    m = df.merge(pred[["frame", "person_id", "pred_group_id"]], on=["frame", "person_id"])
    rows = []
    for f, g in m.groupby("frame"):
        pid, rg, pg = g.person_id.values, g.ref_group_id.values, g.pred_group_id.values
        for i, j in combinations(range(len(g)), 2):
            rows.append((f, pid[i], pid[j],
                         rg[i] != UNLABELLED and rg[i] == rg[j],   # ref: same dataset group
                         pg[i] != -1 and pg[i] == pg[j]))          # pred: same cluster
    return pd.DataFrame(rows, columns=["frame", "a", "b", "ref", "pred"])

def prf(pairs):
    tp = int((pairs.ref & pairs.pred).sum()); fp = int((~pairs.ref & pairs.pred).sum())
    fn = int((pairs.ref & ~pairs.pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return dict(TP=tp, FP=fp, FN=fn, precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.0)

def time_split(df, dev_frac=0.6):
    """Time-block split (weaker than a recording-level split; seq_hotel is one recording).
    Pairs are only evaluated inside a block, so a pair never straddles the boundary."""
    cut = np.sort(df.frame.unique())[int(dev_frac * df.frame.nunique())]
    return df[df.frame < cut], df[df.frame >= cut]
