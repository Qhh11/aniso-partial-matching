"""Target geodesic error normalized by square root of raw target surface area."""
import numpy as np


def geodesic_errors(prediction, references, distances, area):
    p, f = references
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Target area must be positive')
    prediction = np.asarray(prediction, dtype=np.int64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError('Target distances must be a square matrix')
    if prediction.min() < 0 or prediction.max() >= len(distances):
        raise ValueError('Invalid predicted target vertex')
    errors = distances[f, prediction[p]] / np.sqrt(area)
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError('Evaluation distances contain invalid entries')
    return errors


def summarize(errors):
    e = np.asarray(errors)
    if not e.size or not np.isfinite(e).all():
        raise ValueError('Expected nonempty finite errors')
    return dict(mean=float(e.mean()), p05=float((e <= .05).mean()),
                p10=float((e <= .10).mean()), over20=float((e > .20).mean()), samples=int(e.size))


def pck(errors, thresholds):
    sorted_errors = np.sort(np.asarray(errors).reshape(-1))
    if not len(sorted_errors):
        raise ValueError('Cannot evaluate an empty error array')
    return np.searchsorted(sorted_errors, thresholds, side='right') / len(sorted_errors)
