# scripts/train_dsec_cfc.py
"""
Training demo: DSEC -> EventEncoder -> CfC -> classifier
Drop-in script. Edit paths / hyperparams at the top.
"""


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # to avoid potential macOS issues with OpenMP
os.environ['HDF5_PLUGIN_PATH'] = '/Users/sumanthjain/miniforge3/envs/LNN/lib/python3.11/site-packages/hdf5/plugin'
import math
from pathlib import Path
import argparse
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '../src')
from data.event_dataset import DSECEventDataset, collate_events, EventEncoder
from core.cfc_modified import CfCCell
'''
# Try to import DSECEventDataset & collate_events from your data module
try:
    from data.event_dataset import DSECEventDataset, collate_events, EventEncoder
except Exception as e:
    raise ImportError("Cannot import DSECEventDataset/collate_events from src.data.event_dataset. "
                      "Make sure you added the DSEC dataset code to src/data/event_dataset.py") from e

# Try to import CfCCell from your codebase; otherwise use fallback below.
try:
    # adjust path to where your CfC implementation lives
    from core.cfc_modified import CfCCell  # expected to exist in your codebase
    print("Using CfCCell from src.models.cfc")
    CfC_AVAILABLE = True
except Exception:
    CfC_AVAILABLE = False
    print("CfCCell not found in src.models.cfc — using fallback CfC implementation (Eq.10-style).")
'''
'''
# ---------------------
# Fallback CfC cell (simple, matches Eq.10 semantics)
# ---------------------
class FallbackCfCCell(nn.Module):
    """
    Minimal CfC-like cell implementing the closed-form update:
    x_next = sigma(-f * effective_dt) * g + (1 - sigma(-f * effective_dt)) * h

    f, g, h are small MLP heads that take [h_prev, input_v] as input and produce vectors of size hidden_dim.
    effective_dt = dt / softplus(tau)  (tau is a learnable positive scalar per hidden unit)
    This is elementwise and supports batched inputs.
    """
    def __init__(self, input_dim: int, hidden_dim: int, hidden_internal: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        total_in = input_dim + hidden_dim
        # small backbones (linear -> nonlinearity -> linear)
        self.f = nn.Sequential(nn.Linear(total_in, hidden_internal), nn.Tanh(), nn.Linear(hidden_internal, hidden_dim))
        self.g = nn.Sequential(nn.Linear(total_in, hidden_internal), nn.SiLU(), nn.Linear(hidden_internal, hidden_dim))
        self.h = nn.Sequential(nn.Linear(total_in, hidden_internal), nn.SiLU(), nn.Linear(hidden_internal, hidden_dim))

        # softplus-parameterized tau per hidden dim
        self._tau_param = nn.Parameter(torch.ones(hidden_dim) * 0.5)  # initial bias
        self.softplus = nn.Softplus()

    def forward(self, input_v: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor):
        """
        input_v: (B, D)
        h_prev:  (B, H)
        dt:      (B,) or (B,1)
        returns: h_next (B, H)
        """
        if dt.ndim == 1:
            dt = dt.unsqueeze(-1)  # (B,1)
        B = input_v.shape[0]
        # concat along feature dim
        x = torch.cat([h_prev, input_v], dim=-1)  # (B, H + D)
        f_out = self.f(x)  # (B, H)
        g_out = self.g(x)
        h_cand = self.h(x)

        tau = self.softplus(self._tau_param).unsqueeze(0)  # (1, H)
        eff_t = dt / (tau + 1e-12)  # (B, 1) / (1, H) -> broadcast to (B, H)

        gate = torch.sigmoid(-f_out * eff_t)  # (B, H) elementwise
        next_h = gate * g_out + (1.0 - gate) * h_cand
        return next_h


# ---------------------
# Encoder
# ---------------------
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

    def forward(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, dt_feat: torch.Tensor = None):
        # x,y: (N,) pixel coords ; p: (N,) in {-1,+1} or {0,1}
        pos = torch.stack([x.float() / (self.W - 1), y.float() / (self.H - 1)], dim=-1)
        pe = self.pos_proj(pos)
        p_idx = ((p > 0).long()).clamp(0, 1)
        pe_pol = self.pol_emb(p_idx)
        v = self.final(torch.cat([pe, pe_pol], dim=-1))
        if dt_feat is not None:
            v = torch.cat([v, dt_feat.unsqueeze(-1)], dim=-1)
        return v

'''
# ---------------------
# Model wrapper (encoder + CfC + head)
# ---------------------
class EventCfCModel(nn.Module):
    def __init__(self, encoder: nn.Module, cfc_cell: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.cfc = cfc_cell
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_pad, y_pad, p_pad, dt_pad, seq_lens):
        """
        Inputs (padded):
          x_pad, y_pad: (B, T) ints, values -1 in padding positions
          p_pad: (B, T) ints
          dt_pad: (B, T) floats
          seq_lens: (B,) lengths
        Output:
          logits: (B, num_classes)
        """
        device = x_pad.device
        B, T = x_pad.shape
        # build mask
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        mask = idxs < seq_lens.unsqueeze(1)  # (B,T) boolean

        # Flatten and select valid events for encoding
        flat_mask = mask.view(-1)  # (B*T,)
        x_flat = x_pad.view(-1)
        y_flat = y_pad.view(-1)
        p_flat = p_pad.view(-1)
        dt_flat = dt_pad.view(-1)

        valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            # no events in batch (degenerate), return zeros
            h0 = torch.zeros(B, self.hidden_dim, device=device)
            return self.head(h0)

        x_valid = x_flat[valid_idx]
        y_valid = y_flat[valid_idx]
        p_valid = p_flat[valid_idx]
        dt_valid = dt_flat[valid_idx]

        # dt feature: log(dt + eps)
        dt_feat = torch.log(dt_valid + 1e-6)

        # Encode only valid events
        v_valid = self.encoder(x_valid, y_valid, p_valid, dt_feat=dt_feat)  # (N_valid, Denc)
        Denc = v_valid.shape[1]

        # Scatter back into padded v tensor
        v_pad = torch.zeros(B * T, Denc, device=device, dtype=v_valid.dtype)
        v_pad[valid_idx] = v_valid
        v_pad = v_pad.view(B, T, Denc)  # (B, T, Denc)

        # Unroll CfC over time (vectorized over batch)
        h = torch.zeros(B, self.hidden_dim, device=device)
        for t in range(T):
            v_t = v_pad[:, t, :]          # (B, Denc)
            dt_t = dt_pad[:, t]          # (B,)
            # for padded positions, set input to zeros; but mask used to avoid updating? We'll update all and rely on dt small/clamped
            # Optionally skip updates for padded by using mask
            valid_t = mask[:, t]
            if valid_t.any():
                # compute next for all and then choose where to update
                h_next = self.cfc(v_t, h, dt_t)
                # update only where valid
                h = torch.where(valid_t.unsqueeze(-1), h_next, h)
            # else: no valid events for any batch at this timestep -> skip
        logits = self.head(h)
        return logits
    
    def debug_forward(self, x_pad, y_pad, p_pad, dt_pad, seq_lens, max_timesteps_show=8):
        """
        Run forward but collect diagnostics. Returns a dict with:
          - 'v_stats': (mean,std,min,max) for encoded valid vectors
          - 'dt_stats': (mean,std,min,max) for valid dt
          - 'h_norms': list of L2 norms of hidden state after each timestep (first max_timesteps_show steps)
          - 'logits_stats': stats of final logits (mean,std,min,max)
          - 'seq_lens': seq_lens tensor
        """
        device = x_pad.device
        B, T = x_pad.shape
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        mask = idxs < seq_lens.unsqueeze(1)

        flat_mask = mask.view(-1)
        x_flat = x_pad.view(-1)
        y_flat = y_pad.view(-1)
        p_flat = p_pad.view(-1)
        dt_flat = dt_pad.view(-1)

        valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(1)
        out = {}
        if valid_idx.numel() == 0:
            out['error'] = 'no_valid_events'
            out['seq_lens'] = seq_lens.cpu().tolist()
            return out

        x_valid = x_flat[valid_idx]
        y_valid = y_flat[valid_idx]
        p_valid = p_flat[valid_idx]
        dt_valid = dt_flat[valid_idx]

        dt_feat = torch.log(dt_valid + 1e-6)
        v_valid = self.encoder(x_valid, y_valid, p_valid, dt_feat=dt_feat)  # (N_valid, Denc)

        # stats for v_valid and dt_valid
        vt = v_valid.detach()
        out['v_stats'] = {
            'mean': float(vt.mean().cpu()),
            'std' : float(vt.std().cpu()),
            'min' : float(vt.min().cpu()),
            'max' : float(vt.max().cpu()),
            'shape': list(vt.shape)
        }
        dt_t = dt_valid.detach()
        out['dt_stats'] = {
            'mean': float(dt_t.mean().cpu()),
            'std' : float(dt_t.std().cpu()),
            'min' : float(dt_t.min().cpu()),
            'max' : float(dt_t.max().cpu()),
            'count': int(dt_t.numel())
        }

        # scatter back to padded v
        Denc = v_valid.shape[1]
        v_pad = torch.zeros(B * T, Denc, device=device, dtype=v_valid.dtype)
        v_pad[valid_idx] = v_valid
        v_pad = v_pad.view(B, T, Denc)

        # Unroll and collect h norms for first few timesteps
        h = torch.zeros(B, self.hidden_dim, device=device)
        h_norms = []
        steps_to_show = min(T, max_timesteps_show)
        for t in range(T):
            v_t = v_pad[:, t, :]
            dt_t = dt_pad[:, t]
            valid_t = mask[:, t]
            if valid_t.any():
                h_next = self.cfc(v_t, h, dt_t)
                h = torch.where(valid_t.unsqueeze(-1), h_next, h)
            if t < steps_to_show:
                # compute L2 norm per batch and store summary stats
                norms = h.detach().norm(dim=-1)  # (B,)
                h_norms.append({
                    't': t,
                    'mean': float(norms.mean().cpu()),
                    'std' : float(norms.std().cpu()),
                    'min' : float(norms.min().cpu()),
                    'max' : float(norms.max().cpu()),
                })
        logits = self.head(h)
        lt = logits.detach()
        out['h_norms'] = h_norms
        out['logits_stats'] = {
            'mean': float(lt.mean().cpu()),
            'std' : float(lt.std().cpu()),
            'min' : float(lt.min().cpu()),
            'max' : float(lt.max().cpu()),
            'shape': list(lt.shape)
        }
        out['seq_lens'] = seq_lens.cpu().tolist()
        return out




# ---------------------
# Training & evaluation
# ---------------------
def train_loop(model, dataloader, optimizer, criterion, device, epoch, log_step=20, max_grad_norm=5.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()
    for batch_idx, batch in enumerate(dataloader):
        # batch: either encoded or raw; collate_events returns raw x,y,p,dt in DSEC mode
        x = batch['x'].to(device)
        print(x)
        y = batch['y'].to(device)
        print(y)
        p = batch['p'].to(device)
        print(p)
        dt = batch['dt'].to(device)
        print(dt)
        seq_lens = batch['seq_lens'].to(device)
        labels = batch.get('label', None)
        # some datasets may not have labels; for this demo we create dummy labels (e.g., 11 classes)
        if labels is None:
            # create synthetic labels for demo (not for final training)
            labels = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        else:
            labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(x, y, p, dt, seq_lens)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % log_step == 0:
            avg = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / max(1, total)
            print(f"[Epoch {epoch}] Batch {batch_idx+1}/{len(dataloader)}  loss={avg:.4f}  acc={acc:.2f}%")

    dur = time.time() - t0
    return total_loss / max(1, len(dataloader)), 100.0 * correct / max(1, total), dur


def eval_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            p = batch['p'].to(device)
            dt = batch['dt'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            labels = batch.get('label', None)
            if labels is None:
                labels = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            else:
                labels = labels.to(device)

            logits = model(x, y, p, dt, seq_lens)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(1, len(dataloader))
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


# ---------------------
# Main: args & setup
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsec_root", type=str, required=True, help="Path to DSEC sequences folder or file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--enc_dim", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--window_ms", type=int, default=500)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for first few batches")
    parser.add_argument("--dump_tensors", action="store_true", help="Dump one debug batch to debug_batch.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset: raw mode (encoder not provided in dataset) so encoder gradients train
    ds_kwargs = {
        'root': args.dsec_root,
        'sample_mode': 'window',
        'window_ms': args.window_ms,
        'encoder': None,     # encoder in model (trainable)
        'rectify': False,
        'max_events': None
    }
    dataset = DSECEventDataset(**ds_kwargs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_events, drop_last=True)

    # model components
    H_img = 480; W_img = 640
    encoder = EventEncoder(H=H_img, W=W_img, out_dim=args.enc_dim)
    '''
    import sys
    sys.path.insert(0, '../src')

    if CfC_AVAILABLE:
        # try to import and instantiate your CfCCell
        try:
            from core.cfc_modified import CfCCell as RepoCfCCell
            cfc_cell = RepoCfCCell(input_dim=args.enc_dim, hidden_dim=args.hidden_dim)
        except Exception:
            print("Repo CfCCell import failed at instantiation; using fallback.")
            cfc_cell = FallbackCfCCell(input_dim=args.enc_dim, hidden_dim=args.hidden_dim)
    else:
        cfc_cell = FallbackCfCCell(input_dim=args.enc_dim, hidden_dim=args.hidden_dim)
    '''
    cfc_cell = CfCCell(input_size=args.enc_dim, hidden_size=args.hidden_dim)

    model = EventCfCModel(encoder=encoder, cfc_cell=cfc_cell, hidden_dim=args.hidden_dim, num_classes=args.num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, dur = train_loop(model, dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch} TRAIN loss={train_loss:.4f} acc={train_acc:.2f}%  time={dur:.1f}s")
        
        # quick eval on a few batches (no dedicated val set here)
        val_loss, val_acc = eval_loop(model, dataloader, criterion, device)
        print(f"Epoch {epoch}  EVAL loss={val_loss:.4f} acc={val_acc:.2f}%")
        

        # checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict()
        }
        ckpt_path = Path("checkpoints")
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, ckpt_path / f"cfc_dsec_epoch{epoch}.pt")
        #if val_acc > best_val_acc:
        #    best_val_acc = val_acc
        #    torch.save(ckpt, ckpt_path / "cfc_dsec_best.pt")
    #print("Training finished. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
