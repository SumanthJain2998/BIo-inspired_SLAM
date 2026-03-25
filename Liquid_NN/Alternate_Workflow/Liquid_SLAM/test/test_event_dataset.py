import os
import tempfile
import numpy as np
import h5py
import shutil
import pytest

import sys
sys.path.insert(0, '../Liquid_SLAM')
# Adjust this import path if your project uses a different location
from src.data.event_dataset import DSECEventDataset

EPS_CLIP = 1e-9  # matches dataset clipping

def make_events_h5(path, x, y, t, p, ms_to_idx=None, t_offset=None):
    """Create an HDF5 file with DSEC-like structure at `path`."""
    with h5py.File(path, 'w') as f:
        grp = f.create_group('events')
        grp.create_dataset('x', data=np.asarray(x, dtype=np.int32))
        grp.create_dataset('y', data=np.asarray(y, dtype=np.int32))
        grp.create_dataset('t', data=np.asarray(t, dtype=np.int64))  # microseconds
        grp.create_dataset('p', data=np.asarray(p, dtype=np.int8))
        if ms_to_idx is not None:
            grp.create_dataset('ms_to_idx', data=np.asarray(ms_to_idx, dtype=np.int64))
        # t_offset maybe as dataset at root or attribute; the loader checks both - be defensive
        if t_offset is not None:
            # store also as a dataset at root to match loader checks
            f.create_dataset('t_offset', data=np.int64(t_offset))
            # also set group attr for safety
            grp.attrs['t_offset'] = np.int64(t_offset)

def make_rectify_map(path, H, W, offset=(1, 1)):
    """Create rectify_maps.h5 with a simple grid that shifts coords by offset."""
    with h5py.File(path, 'w') as f:
        # rectify_map shape (H, W, 2)
        arr = np.zeros((H, W, 2), dtype=np.float32)
        for yy in range(H):
            for xx in range(W):
                arr[yy, xx, 0] = xx + offset[0]  # new x
                arr[yy, xx, 1] = yy + offset[1]  # new y
        f.create_dataset('rectify_map', data=arr)

def test_basic_read_and_dt_and_polarity(tmp_path):
    # Build a small events file
    x = np.array([0, 10, 20], dtype=np.int32)
    y = np.array([0, 0, 0], dtype=np.int32)
    # microseconds (0, 1e6, 3e6)
    t = np.array([0, 1_000_000, 3_000_000], dtype=np.int64)
    # polarity stored as 0/1 to test mapping
    p = np.array([0, 1, 1], dtype=np.int8)
    t_offset = 1_000_000  # microseconds

    h5file = tmp_path / "events.h5"
    make_events_h5(str(h5file), x, y, t, p, ms_to_idx=None, t_offset=t_offset)

    ds = DSECEventDataset(root=str(tmp_path), sample_mode='full', encoder=None, rectify=False)
    assert len(ds) == 1

    sample = ds[0]
    # Expect t_s = (t + t_offset) * 1e-6
    expected_ts = (t + t_offset).astype(np.float64) * 1e-6
    np.testing.assert_allclose(sample['t'], expected_ts.astype(np.float32), rtol=0, atol=1e-7)

    # dt expected: [eps, 1.0, 2.0] seconds (differences of 1s and 2s)
    expected_dt = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    expected_dt[0] = EPS_CLIP  # dataset clips lower bound to eps
    # dataset converts dt to float32 and then clips
    np.testing.assert_allclose(sample['dt'], expected_dt, rtol=0, atol=1e-7)

    # Polarity mapping: input 0->-1, 1->+1
    np.testing.assert_array_equal(sample['p'], np.array([-1, 1, 1], dtype=np.int8))

def test_polarity_preserve_negative_positive(tmp_path):
    # p already in {-1, +1} should be preserved
    x = np.array([0, 1], dtype=np.int32)
    y = np.array([0, 1], dtype=np.int32)
    t = np.array([0, 500_000], dtype=np.int64)
    p = np.array([-1, 1], dtype=np.int8)
    make_events_h5(str(tmp_path / "events.h5"), x, y, t, p, t_offset=None)

    ds = DSECEventDataset(root=str(tmp_path), sample_mode='full', encoder=None)
    sample = ds[0]
    np.testing.assert_array_equal(sample['p'], p)

def test_window_mode_uses_ms_to_idx(tmp_path):
    # Create many events and a ms_to_idx mapping such that ms_to_idx[ms] = ms*2 (simple)
    N = 2000
    x = np.arange(N, dtype=np.int32) % 5
    y = np.arange(N, dtype=np.int32) % 4
    # times distributed to [0..1999] ms each 1000 microsec multiplier to make indices simple
    # We create t so that ms bin i maps to indices [2*i, 2*i+1] and total_ms = 1000
    # Let t array length = 2000 so ms_to_idx length = 1001 -> total_ms = 1000
    t = (np.arange(N, dtype=np.int64) * 1000)  # in microseconds
    p = np.ones(N, dtype=np.int8)

    # create ms_to_idx such that ms_to_idx[i] = 2*i
    total_ms = 1000
    ms_to_idx = np.zeros(total_ms + 1, dtype=np.int64)
    for i in range(total_ms + 1):
        ms_to_idx[i] = min(2 * i, N)

    make_events_h5(str(tmp_path / "events.h5"), x, y, t, p, ms_to_idx=ms_to_idx, t_offset=None)

    # If window_ms == total_ms, max_start = 0 so start_ms chosen will be 0 deterministically
    ds = DSECEventDataset(root=str(tmp_path), sample_mode='window', window_ms=total_ms, encoder=None)
    sample = ds[0]
    # Expected slice is start_idx = ms_to_idx[0]=0, end_idx = ms_to_idx[total_ms] (<=N)
    expected_start = int(ms_to_idx[0])
    expected_end = int(ms_to_idx[total_ms])
    # The dataset returns the slice; so number of events should be expected_end - expected_start
    assert sample['dt'].shape[0] == (expected_end - expected_start)

def test_rectify_applied_when_requested(tmp_path):
    # small 3x3 grid and a single event at (1,1) should be shifted by offset (1,1) -> new coords (2,2)
    x = np.array([1], dtype=np.int32)
    y = np.array([1], dtype=np.int32)
    t = np.array([0], dtype=np.int64)
    p = np.array([1], dtype=np.int8)
    make_events_h5(str(tmp_path / "events.h5"), x, y, t, p, t_offset=None)
    # create rectify_maps.h5 next to events.h5
    rectify_path = tmp_path / "rectify_maps.h5"
    make_rectify_map(str(rectify_path), H=3, W=3, offset=(1,1))
    ds = DSECEventDataset(root=str(tmp_path), sample_mode='full', encoder=None, rectify=True)
    sample = ds[0]
    # coordinates should be clipped/rounded to integers
    assert sample['x'].shape[0] == 1
    assert sample['y'].shape[0] == 1
    assert sample['x'][0] == 2
    assert sample['y'][0] == 2

def test_empty_events(tmp_path):
    # Make an events file with zero-length arrays
    make_events_h5(str(tmp_path / "events.h5"),
                  x=np.array([], dtype=np.int32),
                  y=np.array([], dtype=np.int32),
                  t=np.array([], dtype=np.int64),
                  p=np.array([], dtype=np.int8),
                  t_offset=None)
    ds = DSECEventDataset(root=str(tmp_path), sample_mode='full', encoder=None)
    sample = ds[0]
    # empty fields
    assert sample['dt'].size == 0
    assert sample['t'].size == 0
    assert sample['p'].size == 0
