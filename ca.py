"""Small cellular automaton: floor-field evacuation, optional group cohesion.
Rules (fixed across scenarios): one ped per cell; 8-neighbour moves toward exit by BFS distance;
a waiting ped gives up after 5 consecutive waits (breaks circular waits); cohesion is released within 3 cells of the exit (otherwise waiting leaders gridlock the door); conflicts resolved by random sequential update order (seeded); peds are removed at the exit.
Only factor changed between scenarios: group membership -> cohesion (leader waits if mate > R cells away)."""
import numpy as np, pandas as pd
from collections import deque

GRID = """\
####################
#..................#
#..................#
#..................#
#.....###..........#
#.....###..........#
#..................#
#..................#
#..................E
#..................#
#..................#
####################"""
MOVES = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

def parse(grid=GRID):
    g = [list(r) for r in grid.splitlines()]
    H, W = len(g), len(g[0]); wall = np.array([[c == "#" for c in r] for r in g])
    exit_ = [(x, y) for y, r in enumerate(g) for x, c in enumerate(r) if c == "E"][0]
    return wall, exit_

def floor_field(wall, exit_):
    """BFS (8-neighbour, diagonals cost 1, no corner cutting) distance to exit."""
    H, W = wall.shape; d = np.full((H, W), 10**6); d[exit_[1], exit_[0]] = 0; q = deque([exit_])
    while q:
        x, y = q.popleft()
        for dx, dy in MOVES:
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and not wall[ny, nx] and d[ny, nx] > d[y, x]+1 \
               and not (wall[y, nx] or wall[ny, x]):
                d[ny, nx] = d[y, x]+1; q.append((nx, ny))
    return d

def place_from_tracks(df, frame, wall, margin=1):
    """Map real positions at one frame to free grid cells (normalised to the interior); nearest free cell on clash."""
    g = df[df.frame == frame]; H, W = wall.shape
    xy = g[["x", "y"]].values; lo, hi = xy.min(0), xy.max(0)
    cells = (xy - lo) / np.maximum(hi - lo, 1e-9) * [W-2-2*margin, H-2-2*margin] + 1 + margin
    taken, pos = set(), {}
    for pid, (cx, cy) in zip(g.person_id.values, np.rint(cells).astype(int)):
        cand = sorted(((x, y) for y in range(H) for x in range(W) if not wall[y, x] and (x, y) not in taken
                       and x < W-3), key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)   # keep clear of exit column
        pos[pid] = cand[0]; taken.add(cand[0])
    return pos

def simulate(pos0, groups, wall, exit_, seed=0, cohesion=True, R=2, release=3, max_wait=5, max_steps=400, speeds=None, stats=None, ref=None):
    """pos0: {pid:(x,y)}, groups: {pid: group_id or None}. Returns (log DataFrame, exit_step dict)."""
    rng = np.random.default_rng(seed); d = floor_field(wall, exit_)
    pos = dict(pos0); H, W = wall.shape; exit_step = {}; log = []
    speeds = speeds or {p: 0.6 + 0.4*rng.random() for p in pos}   # P(move) per step: heterogeneous speeds
    mates = {p: [q for q in pos if q != p and groups.get(p) is not None and groups.get(q) == groups[p]] for p in pos}
    waited = {p: 0 for p in pos}
    for step in range(max_steps):
        for p in rng.permutation(list(pos)):
            if p not in pos: continue
            x, y = pos[p]
            if rng.random() > speeds[p]: continue
            if cohesion and d[y, x] > release:   # (no waiting within `release` cells of exit: avoids gridlock at the door)
                # wait if I'm ahead of a group-mate who is more than R cells behind
                far = [q for q in mates[p] if q in pos and max(abs(pos[q][0]-x), abs(pos[q][1]-y)) > R
                       and d[pos[q][1], pos[q][0]] > d[y, x]]
                if far and waited[p] < max_wait:
                    waited[p] += 1
                    if stats is not None:   # instrumentation only; does not change dynamics
                        stats["waits"] = stats.get("waits", 0) + 1
                        if ref is not None and (ref.get(p) is None or ref.get(p) != ref.get(far[0])):
                            stats["false_waits"] = stats.get("false_waits", 0) + 1   # wait caused by a non-reference group
                    continue
            waited[p] = 0
            occ = set(pos.values()); best, bd = (x, y), d[y, x]
            for dx, dy in MOVES:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < W and 0 <= ny < H) or wall[ny, nx] or wall[y, nx] or wall[ny, x]: continue
                if (nx, ny) in occ: continue
                if d[ny, nx] < bd: best, bd = (nx, ny), d[ny, nx]
            pos[p] = best
            if best == exit_: del pos[p]; exit_step[p] = step+1
        for p, (x, y) in pos.items(): log.append((step, p, x, y, groups.get(p)))
        if not pos: break
    return pd.DataFrame(log, columns=["step", "person_id", "cell_x", "cell_y", "group_id"]), exit_step

def metrics(exit_step, groups, n):
    t = np.array(list(exit_step.values()))
    gids = {g for g in groups.values() if g is not None}
    spread = [np.ptp([exit_step[p] for p in exit_step if groups.get(p) == g]) for g in gids
              if sum(groups.get(p) == g for p in exit_step) > 1]
    return dict(evac_time=max(t) if len(t) else np.nan, mean_exit=t.mean(), exited=len(t)/n,
                group_exit_gap=float(np.mean(spread)) if spread else np.nan)
