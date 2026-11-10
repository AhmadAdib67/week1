"""Proximity-only baseline: per-frame DBSCAN on (x,y) in metres."""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN  # pyright: ignore[reportMissingModuleSource]

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
