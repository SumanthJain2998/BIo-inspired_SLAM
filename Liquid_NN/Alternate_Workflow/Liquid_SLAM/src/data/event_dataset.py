# Add this to your src/data/event_dataset.py (or import it from a new file)

import os
from pathlib import Path
import warnings
from typing import Optional, Callable, List, Dict, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- If you already have EventEncoder from prior message, import or paste it here ---
# Minimal EventEncoder (copy/paste the earlier version or your preferred encoder)
import torch.nn as nn

class EventEncoder(nn.Module):
    def __init__(self, H: int, W: int, d_pos: int = 32, d_pol: int = 8, out_dim: int = 64):
        super().__init__()
        self.H = H
        self.W = W
        self.pos_proj = nn.Linear(2, d_pos)
        self.pol_emb = nn.Embedding(2, d_pol)
        self.final = nn.Sequential(
            nn.LayerNorm(d_pos + d_pol),
            nn.Linear(d_pos + d_pol, out_dim),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, dt_feat: Optional[torch.Tensor] = None):
        # x,y in pixel coords (ints/floats), p in {-1,+1} or {0,1}
        pos = torch.stack([x.float() / (self.W - 1), y.float() / (self.H - 1)], dim=-1)
        pe = self.pos_proj(pos)
        p_idx = ((p > 0).long()).clamp(0, 1)
        pe_pol = self.pol_emb(p_idx)
        v = self.final(torch.cat([pe, pe_pol], dim=-1))
        if dt_feat is not None:
            v = torch.cat([v, dt_feat.unsqueeze(-1)], dim=-1)
        return v


# --- DSECEventDataset ---
class DSECEventDataset(Dataset):
    """
    DSEC event dataset reader.

    Args:
        root: path to DSEC root (folder containing sequence directories or .h5 files)
        split: subfolder name (if applicable) or "all"
        sample_mode: 'full' returns entire file events; 'window' returns a random window of events
        window_ms: if sample_mode == 'window', length of window in milliseconds
        encoder: optional EventEncoder module to apply to raw events on-the-fly (if provided, dataset returns 'v' and 'dt' tensors)
        rectify: bool, whether to attempt rectification using a 'rectify_maps.h5' located next to the events file
        max_events: optional cap to limit number of events returned (for very long sequences)
    """
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "all",
        sample_mode: str = "full",
        window_ms: int = 1000,
        encoder: Optional[nn.Module] = None,
        rectify: bool = False,
        max_events: Optional[int] = None
    ):
        self.root = Path(root)
        self.split = split
        self.sample_mode = sample_mode
        self.window_ms = int(window_ms)
        self.encoder = encoder
        self.rectify = rectify
        self.max_events = max_events

        # find candidate .h5 event files
        self.files = self._find_h5_files()
        if len(self.files) == 0:
            warnings.warn(f"No DSEC .h5 event files found under {self.root}")

    def _find_h5_files(self) -> List[Path]:
        # Look for events.h5 files or any .h5 in structure; allow directory input
        if self.root.is_file() and self.root.suffix in ('.h5', '.hdf5'):
            return [self.root]
        else:
            # common layout: root/sequence_name/events.h5 or root/*/events.h5
            files = list(self.root.glob("**/events.h5"))
            if not files:
                # fallback: any .h5 files
                files = list(self.root.glob("**/*.h5"))
            return files

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f"<DSECEventDataset len={len(self)} sample_mode={self.sample_mode} window_ms={self.window_ms}>"

    def __getitem__(self, idx: int) -> Dict:
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as f:
            # DSEC format: /events/x, /events/y, /events/t, /events/p ; t in microseconds
            if 'events' not in f:
                raise ValueError(f"No 'events' group in {file_path}")

            grp = f['events']

            # --- choose event slice efficiently using ms_to_idx if window mode ---
            if self.sample_mode == 'window' and 'ms_to_idx' in grp:
                ms_to_idx = grp['ms_to_idx'][:]  # numpy array indexed by ms (0..T_ms)
                # total ms range
                total_ms = len(ms_to_idx) - 1
                if total_ms <= 0:
                    # fallback to full
                    start_idx = 0
                    end_idx = grp['x'].shape[0]
                else:
                    # pick a random start ms such that start + window_ms <= total_ms
                    max_start = max(0, total_ms - self.window_ms)
                    start_ms = np.random.randint(0, max_start + 1)
                    end_ms = start_ms + self.window_ms
                    # clamp
                    end_ms = min(end_ms, total_ms)
                    # convert to event indices via ms_to_idx
                    start_idx = int(ms_to_idx[start_ms])
                    end_idx = int(ms_to_idx[end_ms]) if end_ms < len(ms_to_idx) else grp['x'].shape[0]
            else:
                # return full file
                start_idx = 0
                end_idx = grp['x'].shape[0]

            # Optionally limit by max_events
            if self.max_events is not None and (end_idx - start_idx) > self.max_events:
                end_idx = start_idx + self.max_events

            # Read slice (h5py supports slicing without loading whole arrays)
            x = grp['x'][start_idx:end_idx].astype(np.int32)
            y = grp['y'][start_idx:end_idx].astype(np.int32)
            t = grp['t'][start_idx:end_idx].astype(np.int64)  # microseconds
            p = grp['p'][start_idx:end_idx].astype(np.int8)

            # Add t_offset if provided (microseconds)
            t_offset = grp.attrs.get('t_offset', None)
            # Note: in some DSEC files the t_offset is located at /t_offset path (not an attr)
            if t_offset is None:
                if 't_offset' in f:
                    t_offset = f['t_offset'][()]  # dataset or attribute
                elif 't_offset' in grp:
                    t_offset = grp['t_offset'][()]
            if t_offset is None:
                t_offset = 0
            t = t + int(t_offset)

            # rectify coordinates if requested and rectify map available
            if self.rectify:
                # Typical rectify map is stored in a sibling file 'rectify_maps.h5'
                rect_path = file_path.parent / 'rectify_maps.h5'
                if rect_path.exists():
                    with h5py.File(rect_path, 'r') as rf:
                        # rectify_map shape: (H, W, 2) or similar
                        rectify_map_ds = rf.get('rectify_map', None)
                        if rectify_map_ds is not None:
                            # load rectify map into memory (num_pixels x 2). This is small for standard sensors (e.g. 640x480x2).
                            rectify_arr = rectify_map_ds[()]  # now a NumPy array of shape (H_map, W_map, 2)
                            H_map, W_map = rectify_arr.shape[:2]

                            # clip indices then index using NumPy advanced indexing
                            x_cl = np.clip(x, 0, W_map - 1)
                            y_cl = np.clip(y, 0, H_map - 1)
                            rect_coords = rectify_arr[y_cl, x_cl]  # (N,2) — now valid because rectify_arr is a NumPy array

                            x = np.round(rect_coords[:, 0]).astype(np.int32)
                            y = np.round(rect_coords[:, 1]).astype(np.int32)

                        else:
                            # Some datasets store maps differently; warn and skip
                            warnings.warn(f"Rectify map not found in {rect_path}; skipping rectification.")
                else:
                    warnings.warn(f"Requested rectification but rectify_maps.h5 not found next to {file_path}.")

            # Polarity mapping: DSEC might store p as {0,1} or {-1,1}; map to {-1, +1}
            p_unique = np.unique(p)
            if set(p_unique.tolist()) <= {0, 1}:
                p = np.where(p > 0, 1, -1).astype(np.int8)
            elif set(p_unique.tolist()) <= {-1, 1}:
                p = p.astype(np.int8)
            else:
                # fallback: treat any positive value as +1
                p = np.where(p > 0, 1, -1).astype(np.int8)

            # Convert times to seconds (float)
            t_s = t.astype(np.float64) * 1e-6  # microseconds -> seconds
            if t_s.size == 0:
                # empty sequence
                dt = np.zeros(0, dtype=np.float32)
            else:
                dt = np.concatenate([[0.0], np.diff(t_s)]).astype(np.float32)  # seconds

            # Clip dt numerically to avoid zero/huge values in CfC
            eps = 1e-9
            dt = np.clip(dt, eps, 1e3)  # clamp upper bound (1000s) to avoid insane dt

            # Prepare output
            # Option A: if encoder provided, return encoded v + dt
            if self.encoder is not None:
                # Note: encoder expects tensors on appropriate device; keep on CPU for now (DataLoader will move to device or you can move inside training loop)
                x_t = torch.from_numpy(x).long()
                y_t = torch.from_numpy(y).long()
                p_t = torch.from_numpy(p).long()
                dt_t = torch.from_numpy(dt).float()
                with torch.no_grad():  # encoding performed here without grad (optionally set requires_grad if encoder is part of model)
                    # If you want encoder weights to train, remove no_grad and ensure encoder is in dataset->model flow or perform encoding in model
                    v = self.encoder(x_t, y_t, p_t, dt_feat=torch.log(dt_t + 1e-6))
                # v: (N, D)
                result = {
                    'v': v,                 # torch tensor (N, D)
                    'dt': dt_t,             # torch tensor (N,)
                    't': torch.from_numpy(t_s).float(),
                    'sample_id': str(file_path)
                }
            else:
                # Return raw numpy arrays (caller/transform can encode later)
                result = {
                    'x': x,
                    'y': y,
                    'p': p,
                    't': t_s.astype(np.float32),
                    'dt': dt.astype(np.float32),
                    'sample_id': str(file_path)
                }

            return result


# --- collate function for variable-length event sequences ---
def collate_events(batch: List[Dict]) -> Dict:
    """
    Collate function that pads encoded vectors (v) and dt arrays.
    Expects each item in batch to contain either:
      - 'v' (torch tensor (N_i, D)) and 'dt' (torch tensor (N_i,))
    or
      - raw 'x','y','p','dt' numpy arrays (then we convert to tensors).
    Returns:
      - v: (B, T, D)
      - dt: (B, T)
      - seq_lens: (B,)
      - sample_ids: list
    """
    # detect encoded vs raw
    first = batch[0]
    encoded_mode = 'v' in first

    seq_lens = torch.tensor([item['v'].shape[0] if encoded_mode else len(item['dt']) for item in batch], dtype=torch.long)
    maxL = int(seq_lens.max().item()) if len(seq_lens) > 0 else 0
    B = len(batch)

    if encoded_mode:
        D = batch[0]['v'].shape[1]
        v_pad = torch.zeros(B, maxL, D, dtype=torch.float)
        dt_pad = torch.zeros(B, maxL, dtype=torch.float)
        sample_ids = []
        for i, item in enumerate(batch):
            L = item['v'].shape[0]
            v_pad[i, :L] = item['v']
            dt_pad[i, :L] = item['dt']
            sample_ids.append(item.get('sample_id', ''))
        return {'v': v_pad, 'dt': dt_pad, 'seq_lens': seq_lens, 'sample_ids': sample_ids}
    else:
        # convert raw arrays to tensors (and leave x,y,p in separate tensors if needed)
        # For simplicity return padded dt and keep raw x,y,p lists
        dt_pad = torch.zeros(B, maxL, dtype=torch.float)
        x_pad = torch.full((B, maxL), -1, dtype=torch.long)
        y_pad = torch.full((B, maxL), -1, dtype=torch.long)
        p_pad = torch.zeros(B, maxL, dtype=torch.long)
        sample_ids = []
        for i, item in enumerate(batch):
            L = len(item['dt'])
            dt_pad[i, :L] = torch.from_numpy(item['dt'])
            x_pad[i, :L] = torch.from_numpy(item['x']).long()
            y_pad[i, :L] = torch.from_numpy(item['y']).long()
            p_pad[i, :L] = torch.from_numpy(item['p']).long()
            sample_ids.append(item.get('sample_id', ''))
        return {'x': x_pad, 'y': y_pad, 'p': p_pad, 'dt': dt_pad, 'seq_lens': seq_lens, 'sample_ids': sample_ids}


# --- add DSEC option to your create_event_dataloader factory ---
def create_event_dataloader(
    dataset_type: str,
    dataset_config: Dict,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    datasets = {
        'hdf5': HDF5EventDataset,           # your existing
        'dvs_gesture': DVSGestureDataset,   # your existing
        'synthetic': SyntheticEventDataset, # your existing
        'dsec': DSECEventDataset            # new
    }
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset = datasets[dataset_type](**dataset_config)
    # pick collate depending on encoded/raw
    collate = collate_events

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate
    )
