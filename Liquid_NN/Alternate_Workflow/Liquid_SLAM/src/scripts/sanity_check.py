# scripts/debug_read_dsec_sample.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # to avoid potential macOS issues with OpenMP
os.environ['HDF5_PLUGIN_PATH'] = '/Users/sumanthjain/miniforge3/envs/LNN/lib/python3.11/site-packages/hdf5/plugin'

import sys
sys.path.insert(0, '../src')
from data.event_dataset import DSECEventDataset
import numpy as np



ds = DSECEventDataset(root="/Users/sumanthjain/Liquid Neural networks/Code Base Development/zurich_city_00_a", sample_mode='window', window_ms=500, encoder=None)
sample = ds[0]
print("keys:", sample.keys())
print("n_events:", sample['dt'].shape[0])
print("first 8 x,y,t,p,dt:")
for i in range(min(8, sample['dt'].shape[0])):
    print(i, sample['x'][i], sample['y'][i], sample['t'][i], sample['p'][i], sample['dt'][i])
