"""Intensity display scaling; numerical fits always keep their existing scale."""
import numpy as np


def normalize_catalog(frame, xmin, xmax):
    inside = frame['Freq'].between(xmin, xmax)
    values = frame.loc[inside, 'Intensity'].to_numpy(dtype=float)
    valid = values[np.isfinite(values) & (values > 0)]
    reference = float(valid.max()) if valid.size else None
    # No overlapping positive prediction: no arbitrary out-of-band reference.
    frame['Norm_Intensity'] = frame['Intensity'] / reference if reference else 0.0
    return reference


def scale_figure(fig, factor, label):
    for trace in fig.data:
        if trace.y is not None:
            trace.y = [None if value is None else float(value) * factor for value in trace.y]
    if fig.layout.yaxis.range is not None:
        fig.layout.yaxis.range = [float(v) * factor for v in fig.layout.yaxis.range]
    for shape in fig.layout.shapes or ():
        if (shape.yref or 'y') == 'y':
            for key in ('y0', 'y1'):
                value = getattr(shape, key, None)
                if value is not None:
                    setattr(shape, key, float(value) * factor)
    for annotation in fig.layout.annotations or ():
        if (annotation.yref or 'y') == 'y' and annotation.y is not None:
            annotation.y *= factor
    fig.update_yaxes(title_text=label)
    return fig


def normalized_relayout(relayout, factor):
    data = dict(relayout or {})
    for key in ('yaxis.range[0]', 'yaxis.range[1]'):
        if data.get(key) is not None:
            data[key] = float(data[key]) / factor
    if data.get('yaxis.range') is not None:
        data['yaxis.range'] = [float(v) / factor for v in data['yaxis.range']]
    return data
