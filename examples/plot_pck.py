"""Plot saved threshold,pck CSV files without loading meshes or checkpoints."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('curves', nargs='+', help='One or more threshold,pck CSV files')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--colors', nargs='+')
    parser.add_argument('--linestyles', nargs='+')
    parser.add_argument('--output', default='outputs/pck.png')
    parser.add_argument('--max-error', type=float, default=.5)
    args = parser.parse_args()
    for key in ('labels', 'colors', 'linestyles'):
        value = getattr(args, key)
        if value is not None and len(value) != len(args.curves):
            raise ValueError(f'--{key} requires one entry per curve')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, path in enumerate(args.curves):
        data = np.genfromtxt(path, delimiter=',', names=True)
        options = dict(label=args.labels[i] if args.labels else Path(path).stem)
        if args.colors:
            options['color'] = args.colors[i]
        if args.linestyles:
            options['linestyle'] = args.linestyles[i]
        ax.plot(data['threshold'], 100*data['pck'], **options)
    ax.set(xlabel='Normalized geodesic error', ylabel='PCK (%)',
           xlim=(0, args.max_error), ylim=(0, 100))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
