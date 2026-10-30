"""plotting helpers on the loom style and palette."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

STYLE = Path(__file__).resolve().parent / 'loom.mplstyle'
FIGURES = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
TABLES = FIGURES.parent / 'tables'

PALETTE = ['#bc80bd', '#fb8072', '#b3de69', '#fdb462', '#fccde5', '#8dd3c7',
           '#ffed6f', '#bebada', '#80b1d3', '#ccebc5', '#d9d9d9']
INK, GRID = '#323034', '#b1afb5'
SECTOR = [PALETTE[0], PALETTE[8], PALETTE[2], PALETTE[3]]
SOURCE, OFFICE, HIGHLIGHT, NODE = PALETTE[3], PALETTE[2], PALETTE[1], PALETTE[10]


def setup():
    """apply the loom style; fall back to mathtext when latex is not installed (e.g. colab)."""
    plt.style.use(STYLE)
    try:
        fig = plt.figure()
        fig.text(0.5, 0.5, r'$x$')
        fig.canvas.draw()
    except Exception as exc:
        plt.rcParams['text.usetex'] = False
        print(f'latex unavailable, using mathtext ({type(exc).__name__})')
    finally:
        plt.close('all')


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f'{name}.pdf')


def table(name, rows, header, fmt):
    """write a booktabs-free tabular body for the report: header row, then rows formatted by fmt."""
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [' & '.join(header) + r' \\ \hline']
    lines += [' & '.join(f.format(v) for f, v in zip(fmt, r)) + r' \\' for r in rows]
    (TABLES / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def pipes(ax, net, edges=None, color=GRID, lw=0.8, **kw):
    edges = net.pipes if edges is None else edges
    segs = [(net.coords[e[0]], net.coords[e[1]]) for e in edges]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, **kw))


def route(ax, net, nodes, color=HIGHLIGHT, lw=1.8, **kw):
    xy = [net.coords[u] for u in nodes]
    ax.add_collection(LineCollection(list(zip(xy, xy[1:])), colors=color, linewidths=lw,
                                     capstyle='round', **kw))


def nodes(ax, net, ids=None, color=NODE, s=12, marker='o', zorder=3, **kw):
    ids = net.nodes if ids is None else list(ids)
    xs = [net.coords[u][0] for u in ids]
    ys = [net.coords[u][1] for u in ids]
    return ax.scatter(xs, ys, s=s, c=color, marker=marker, linewidths=0.4, zorder=zorder, **kw)


def sources(ax, net, color=SOURCE, s=70, **kw):
    return nodes(ax, net, net.sources, color=color, s=s, marker='*', zorder=5, **kw)


def office(ax, net, s=34, **kw):
    return nodes(ax, net, [net.office], color=OFFICE, s=s, marker='D', zorder=5, **kw)


def instance_axes(nets, layout=(('NYT', 'PES'), ('HAN', 'FOS')), width=6.27, extra=0.3):
    """one panel per instance, widths proportional to each network's aspect ratio so all
    panels in a row share a height. extra is vertical room (inches) for titles and legend."""
    aspect = {}
    for k in (k for row in layout for k in row):
        x0, x1, y0, y1 = nets[k].bbox()
        aspect[k] = (x1 - x0) / (y1 - y0)
    heights = [width / sum(aspect[k] for k in row) for row in layout]
    fig = plt.figure(figsize=(width, sum(heights) + extra * len(layout)))
    gs = fig.add_gridspec(len(layout), 1, height_ratios=heights, hspace=0.12,
                          left=0.005, right=0.995, top=0.96, bottom=0.01)
    axes = {}
    for r, row in enumerate(layout):
        sub = gs[r].subgridspec(1, len(row), width_ratios=[aspect[k] for k in row], wspace=0.08)
        for c, k in enumerate(row):
            axes[k] = fig.add_subplot(sub[c])
    return fig, axes


def frame(ax, title=None):
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.margins(0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
