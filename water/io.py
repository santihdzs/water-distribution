"""instance loading and validation."""
import math
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
INSTANCES = ('NYT', 'HAN', 'FOS', 'PES')


@dataclass
class Network:
    name: str
    coords: dict                # id -> (x, y)
    sources: list               # source ids, file order
    pipes: list                 # (u, v, capacity), one record per physical pipe
    office: int | None
    new: list                   # accepted new nodes (x, y, capacity)
    header: tuple               # (V, E) as declared in the file
    issues: list = field(default_factory=list)

    @property
    def nodes(self):
        return sorted(self.coords)

    def length(self, u, v):
        return math.dist(self.coords[u], self.coords[v])

    def bbox(self):
        xs, ys = zip(*self.coords.values())
        return min(xs), max(xs), min(ys), max(ys)


def _finite(*vals):
    return all(math.isfinite(v) for v in vals)


def load(name, path=None):
    path = Path(path) if path else DATA / f'{name}.txt'
    lines = [l.strip() for l in path.read_text().splitlines()]
    issues = []
    try:
        header = tuple(int(t) for t in lines[0].split()[:2])
    except (ValueError, IndexError):
        header = (None, None)
        issues.append('header: missing or malformed "V E" line')

    raw = {'[NODES]': [], '[EDGES]': [], '[OFFICE]': [], '[NEW]': []}
    section = None
    for i, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        if line.startswith('['):
            section = line if line in raw else None
            if section is None:
                issues.append(f'line {i}: unknown section {line}, ignored')
            continue
        if section:
            raw[section].append((i, line.split()))

    coords, sources = {}, []
    for i, t in raw['[NODES]']:
        try:
            nid, x, y, flag = int(t[0]), float(t[1]), float(t[2]), int(t[3])
        except (ValueError, IndexError):
            issues.append(f'line {i}: malformed node record, skipped')
            continue
        if not _finite(x, y):
            issues.append(f'line {i}: node {nid} has non-finite coordinates, skipped')
            continue
        if nid in coords:
            issues.append(f'line {i}: duplicate node id {nid}, first record kept')
            continue
        if flag not in (0, 1):
            issues.append(f'line {i}: node {nid} source flag {flag} not 0/1, treated as non-source')
            flag = 0
        coords[nid] = (x, y)
        if flag:
            sources.append(nid)

    pipes, seen = [], {}
    for i, t in raw['[EDGES]']:
        try:
            u, v, c = int(t[0]), int(t[1]), float(t[2])
        except (ValueError, IndexError):
            issues.append(f'line {i}: malformed pipe record, skipped')
            continue
        if u not in coords or v not in coords:
            issues.append(f'line {i}: pipe {u}-{v} references an unknown node, skipped')
            continue
        if u == v:
            issues.append(f'line {i}: self-loop at {u}, skipped')
            continue
        if not _finite(c) or c <= 0:
            issues.append(f'line {i}: pipe {u}-{v} has non-positive capacity {c}, skipped')
            continue
        key = frozenset((u, v))
        if key in seen:
            prev = pipes[seen[key]]
            same = 'same' if prev[2] == c else f'different ({prev[2]} vs {c})'
            issues.append(f'line {i}: pipe {u}-{v} repeats an earlier record with {same} capacity, '
                          f'treated as one pipe (first record kept)')
            continue
        seen[key] = len(pipes)
        pipes.append((u, v, c))

    by_xy = {}
    for nid, xy in coords.items():
        by_xy.setdefault(xy, []).append(nid)
    for xy, ids in by_xy.items():
        if len(ids) > 1:
            issues.append(f'nodes {ids} share coordinates {xy} (zero-length pipes possible)')

    ids = sorted(coords)
    gaps = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids)) if ids else []
    if gaps:
        issues.append(f'node ids are not contiguous ({len(gaps)} gaps, e.g. {gaps[:5]}); '
                      f'handled by mapping ids to indices')
    if header != (len(coords), len(pipes)):
        issues.append(f'header declares V={header[0]}, E={header[1]}; '
                      f'loaded V={len(coords)}, E={len(pipes)}')
    if not sources:
        issues.append('no source nodes')

    office = None
    if raw['[OFFICE]']:
        try:
            office = int(raw['[OFFICE]'][0][1][0])
        except ValueError:
            issues.append('office record malformed')
        if office is not None and office not in coords:
            issues.append(f'office {office} is not a node')
            office = None
    else:
        issues.append('no office node listed')

    net = Network(name, coords, sources, pipes, office, [], header, issues)
    x0, x1, y0, y1 = net.bbox()
    mx, my = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    for i, t in raw['[NEW]']:
        try:
            x, y, c = float(t[0]), float(t[1]), float(t[2])
        except (ValueError, IndexError):
            issues.append(f'line {i}: malformed new-node record, skipped')
            continue
        # a point far outside the drawn network cannot be a local expansion; reject it
        if not (x0 - mx <= x <= x1 + mx and y0 - my <= y <= y1 + my):
            issues.append(f'line {i}: new node ({x:g}, {y:g}) lies outside the network bounding box '
                          f'(+50% margin), rejected as an outlier')
            continue
        net.new.append((x, y, c))
    return net


def load_all():
    return {name: load(name) for name in INSTANCES}
