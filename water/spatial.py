"""nearest non-source node: brute force scan vs a 2-d tree."""
import math


def brute_nearest(points, q):
    """points: dict id -> (x, y). checks every candidate, O(n)."""
    best = min(points, key=lambda u: (math.dist(points[u], q), u))
    return best, math.dist(points[best], q), len(points)


class KDTree:
    """divide and conquer: split on the median of alternating axes.
    nodes are [id, point, axis, left, right]."""

    def __init__(self, points):
        self.root = self._build(sorted(points.items()), 0)

    def _build(self, items, depth):
        if not items:
            return None
        axis = depth % 2
        items.sort(key=lambda it: (it[1][axis], it[0]))
        m = len(items) // 2
        return [items[m][0], items[m][1], axis,
                self._build(items[:m], depth + 1), self._build(items[m + 1:], depth + 1)]

    def insert(self, nid, p):
        """descend to a leaf slot, O(depth). no rebalancing."""
        if self.root is None:
            self.root = [nid, p, 0, None, None]
            return
        node = self.root
        while True:
            side = 3 if p[node[2]] < node[1][node[2]] else 4
            if node[side] is None:
                node[side] = [nid, p, 1 - node[2], None, None]
                return
            node = node[side]

    def nearest(self, q):
        """branch and prune: visit the near child first, the far child only if the
        splitting line is closer than the best distance so far."""
        best = [None, math.inf, 0]

        def visit(node):
            if node is None:
                return
            best[2] += 1
            d = math.dist(node[1], q)
            if d < best[1] or (d == best[1] and node[0] < best[0]):
                best[0], best[1] = node[0], d
            diff = q[node[2]] - node[1][node[2]]
            near, far = (node[3], node[4]) if diff < 0 else (node[4], node[3])
            visit(near)
            if abs(diff) <= best[1]:
                visit(far)

        visit(self.root)
        return tuple(best)

    def depth(self):
        def h(node):
            return 0 if node is None else 1 + max(h(node[3]), h(node[4]))
        return h(self.root)
