"""Proximity-only baseline: per-frame DBSCAN on (x,y) in metres."""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN

def detect_proximity(df, eps=1.0, min_samples=2, config_id="dev"):
    """Cluster each annotated frame independently. pred_group_id = -1 means noise/singleton.
    Output schema = design doc 3.2. Cluster ids are only unique within a frame."""
    out = []
    for f, g in df.groupby("frame"):
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(g[["x", "y"]].values) \
              if len(g) >= min_samples else np.full(len(g), -1)
        out.append(pd.DataFrame({"frame": f, "person_id": g.person_id.values, "pred_group_id": lab}))
    p = pd.concat(out, ignore_index=True)
    p["method"], p["config_id"] = "proximity", f"{config_id}_eps{eps}"
    return p


def detect_motion(df, eps=1.0, cos_min=0.8, v_min=0.3, min_samples=2, config_id="dev"):
    """Proximity + ONE motion-consistency feature: two people may only be linked if both are moving
    (speed >= v_min m/s) and their headings agree (cosine >= cos_min). Incompatible pairs get infinite
    distance in DBSCAN. Consequence (by design): stationary pairs are never grouped.
    Uses dataset-supplied vx, vy. Clusters are still chained transitively by DBSCAN."""
    from scipy.spatial.distance import cdist
    out = []
    for f, g in df.groupby("frame"):
        n = len(g)
        if n < min_samples:
            lab = np.full(n, -1)
        else:
            P, V = g[["x", "y"]].values, g[["vx", "vy"]].values
            sp = np.linalg.norm(V, axis=1); U = V / np.maximum(sp[:, None], 1e-9)
            D = cdist(P, P); ok = (sp[:, None] >= v_min) & (sp[None, :] >= v_min) & (U @ U.T >= cos_min)
            D[~ok] = 1e6; np.fill_diagonal(D, 0)
            lab = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)
        out.append(pd.DataFrame({"frame": f, "person_id": g.person_id.values, "pred_group_id": lab}))
    p = pd.concat(out, ignore_index=True)
    p["method"], p["config_id"] = "motion", f"{config_id}_eps{eps}_cos{cos_min}_v{v_min}"
    return p
