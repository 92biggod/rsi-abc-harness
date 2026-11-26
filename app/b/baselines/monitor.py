"""Describe current vibration distribution changes using labeled, projected splits."""
import json
import numpy as np

def feature_shift(train, validation):
    # Separate a change in typical level from an unusually wide validation range.
    train_quantiles = np.quantile(train['x'], [.1,.5,.9], axis=0)
    valid_quantiles = np.quantile(validation['x'], [.1,.5,.9], axis=0)
    floor = 1e-8 * np.maximum(1., np.abs(train_quantiles[1]))
    width = np.maximum(train_quantiles[2]-train_quantiles[0], floor)
    shifts = (valid_quantiles[1]-train_quantiles[1])/width
    result = []
    for j in np.argsort(-np.abs(shifts),kind='stable')[:2]:
        path = json.loads(train['paths'][j])
        result.append(dict(feature='.'.join(str(step[1]) for step in path),
                           median_shift_in_train_widths=round(float(shifts[j]),3)))
    return result
