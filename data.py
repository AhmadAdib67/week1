"""Data layer: load/validate ETH Hotel obsmat + groups, schema, replay."""
import numpy as np, pandas as pd, os
REQUIRED = ["frame", "person_id", "x", "y"]
UNLABELLED = -1   # "unlabelled-non-group" (NOT a confirmed singleton, see design doc 2.1)

def validate(df, expect_ids=None, expect_rows=None):
    """Schema + data-quality checks. Raises on missing columns; returns a report dict."""
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    rep = {"rows": len(df), "ids": df.person_id.nunique(),
           "duplicate_person_frames": int(df.duplicated(["frame", "person_id"]).sum()),
           "missing_positions": int(df[["x", "y"]].isna().sum().sum()),
           "frames": df.frame.nunique(),
           "median_track_len": float(df.groupby("person_id").size().median())}
    if expect_ids is not None and rep["ids"] != expect_ids: raise ValueError(f"ids {rep['ids']}!={expect_ids}")
    if expect_rows is not None and rep["rows"] != expect_rows: raise ValueError(f"rows {rep['rows']}!={expect_rows}")
    return rep

def load_groups(path):
    """groups.txt -> list of lists of person ids (dataset-provided labels)."""
    return [[int(t) for t in l.split()] for l in open(path) if l.strip()]

def load_eth(data_dir="data"):
    """obsmat.txt -> schema table. x=col3 (pos_x), y=col5 (pos_y); col4 is pos_z (unused)."""
    o = np.loadtxt(os.path.join(data_dir, "obsmat.txt"))
    df = pd.DataFrame({"frame": o[:, 0].astype(int), "person_id": o[:, 1].astype(int),
                       "x": o[:, 2], "y": o[:, 4], "vx": o[:, 5], "vy": o[:, 7],
                       "recording_id": "seq_hotel"})
    groups = load_groups(os.path.join(data_dir, "groups.txt"))
    gid = {p: i for i, g in enumerate(groups) for p in g}
    df["ref_group_id"] = df.person_id.map(gid).fillna(UNLABELLED).astype(int)
    return df.sort_values(["frame", "person_id"]).reset_index(drop=True), groups

def world_to_pixel(xy, H):
    """World metres -> (px, py) in bg.png. H maps pixel(row,col)->world, so invert it."""
    p = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(H).T
    p = p[:, :2] / p[:, 2:]
    return np.c_[p[:, 1], p[:, 0]]   # (col,row) = (x,y) in image

def replay(df, data_dir="data", out="outputs/replay.gif", max_steps=120, start=None):
    """Animated GIF of tracks over the background frame; dataset-labelled group members in colour."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    H = np.loadtxt(os.path.join(data_dir, "H.txt"))
    bg = plt.imread(os.path.join(data_dir, "bg.png"))
    frames = np.sort(df.frame.unique())
    if start is None: start = frames[np.argmax(df.groupby("frame").size().rolling(20).sum().values)-19] if len(frames) > 20 else frames[0]
    frames = frames[frames >= start][:max_steps]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7.2, 5.76)); ax.imshow(bg); ax.axis("off")
    trail = 8
    def draw(i):
        ax.clear(); ax.imshow(bg); ax.axis("off")
        cur = frames[i]
        for pid, g in df[(df.frame <= cur) & (df.frame > cur - 10 * trail)].groupby("person_id"):
            px = world_to_pixel(g[["x", "y"]].values, H)
            gid = g.ref_group_id.iloc[0]
            c, lw = ("cyan", 1) if gid == UNLABELLED else (cmap(gid % 20), 2.5)
            ax.plot(px[:, 0], px[:, 1], "-", color=c, lw=lw)
            if g.frame.iloc[-1] == cur: ax.plot(px[-1, 0], px[-1, 1], "o", color=c, ms=7, mec="k")
        ax.set_title(f"ETH Hotel replay  frame {cur}  (coloured = dataset-labelled group)", fontsize=8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    FuncAnimation(fig, draw, frames=len(frames)).save(out, writer=PillowWriter(fps=5)); plt.close(fig)
    return out
