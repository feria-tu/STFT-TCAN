from utils.parser import *
from utils.folderconstants import *

# Threshold parameters
lm_d = {
    'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
    'NAB': [(0.991, 1), (0.99, 1)],
    'SMAP': [(0.98, 1), (0.98, 1)],
    'MBA': [(0.87, 1), (0.93, 1.04)],
    'SWaT': [(0.993, 1), (0.993, 1)],
    'WADI': [(0.99, 1), (0.999, 1)],
    'MSL': [(0.97, 1), (0.999, 1.04)],
    'UCR': [(0.993, 1), (0.99935, 1)],
}
lm = lm_d[args.dataset][0]

# Hyperparameters lr
lr_d = {
    'SMD': 0.0001,
    'SMAP': 0.001,
    'NAB': 0.009,
    'MBA': 0.001,
    'SWaT': 0.008,
    'WADI': 0.0001,
    'MSL': 0.002,
    'UCR': 0.006,
}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
    'SMD': (98, 2000),
    'SMAP': (97, 5000),
    'NAB': (98, 2),
    'MBA': (99, 2),
    'SWaT': (95, 10),
    'WADI': (99, 1200),
    'MSL': (97, 150),
    'UCR': (98, 2),
}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9
