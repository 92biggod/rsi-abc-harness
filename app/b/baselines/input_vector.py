"""Fast exact-schema path for the published 34-feature B request."""
import json
import math

from .data import flatten

CHANNELS=('horizontal','vertical')
FEATURES=tuple(sorted(('rms','peak','peak_to_peak','mean_abs','kurtosis','skewness','crest_factor',
                       'shape_factor','impulse_factor','margin_factor','spectral_centroid','spectral_rms',
                       'spectral_kurtosis','dominant_freq','band1_energy_0_2k','band2_energy_2k_6k',
                       'band3_energy_6k_12k')))
KEYS=frozenset(FEATURES)
SCHEMA=[json.dumps([['key','features'],['key',ch],['key',f]],separators=(',',':'))
        for ch in CHANNELS for f in FEATURES]


def vectorizer(paths):
    def checked(inputs):
        actual,values=flatten(inputs)
        if actual!=paths:
            raise ValueError('Incompatible B input schema')
        return values
    if paths!=SCHEMA:
        return checked

    def encode(inputs):
        if type(inputs) is not dict or inputs.keys()!={'features'}:
            return checked(inputs)
        features=inputs['features']
        if type(features) is not dict or features.keys()!=set(CHANNELS):
            return checked(inputs)
        values=[]
        for ch in CHANNELS:
            block=features[ch]
            if type(block) is not dict or block.keys()!=KEYS:
                return checked(inputs)
            values.extend(block[k] for k in FEATURES)
        if not all(type(v) in (float,int) and math.isfinite(v) for v in values):
            return checked(inputs)
        return values
    return encode
