# cfc_modified.py
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

_eps = 1e-6

def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    # inverse of softplus: softplus^{-1}(y) = log(exp(y) - 1)
    return torch.log(torch.expm1(x.clamp(min=1e-6)))

class CfCCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "default",
        activation: str = "tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        min_tau: float = 0.05,
        init_tau_low: float = 0.1,
        init_tau_high: float = 10.0,
        readout: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.min_tau = float(min_tau)
        
        # backbone MLP
        self.backbone = self._build_backbone(input_size + hidden_size, backbone_units, backbone_layers, backbone_dropout, activation)
        # three heads: g, f, h
        self.head_g = nn.Linear(backbone_units, hidden_size)
        self.head_f = nn.Linear(backbone_units, hidden_size)
        self.head_h = nn.Linear(backbone_units, hidden_size)
        
        # optional readout projection (maps hidden state to output)
        self.readout = nn.Linear(hidden_size, hidden_size) if readout else None
        
        # initialize tau param via softplus inverse so that tau = min_tau + softplus(tau_param)
        init_tau = torch.empty(hidden_size).uniform_(init_tau_low, init_tau_high)
        init_tau_shifted = init_tau - self.min_tau
        init_tau_shifted = torch.clamp(init_tau_shifted, min=1e-4)
        tau_param_init = inverse_softplus(init_tau_shifted)
        self.tau_param = nn.Parameter(tau_param_init)
        
        self._init_weights()
    
    def _build_backbone(self, input_dim, units, layers, dropout, activation):
        act = self._get_activation(activation)
        modules = []
        in_f = input_dim
        for i in range(layers):
            modules.append(nn.Linear(in_f, units))
            modules.append(act())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            in_f = units
        return nn.Sequential(*modules)
    
    @staticmethod
    def _get_activation(name):
        name = name.lower()
        if name in ("tanh", "lecun_tanh"):
            return nn.Tanh
        if name == "relu":
            return nn.ReLU
        if name == "gelu":
            return nn.GELU
        if name == "sigmoid":
            return nn.Sigmoid
        raise ValueError("Unknown activation")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def tau(self) -> torch.Tensor:
        return self.min_tau + torch.nn.functional.softplus(self.tau_param)
    
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed_time: Optional[torch.Tensor] = None,
        return_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = input.shape[0]
        device = input.device
        dtype = input.dtype
        
        if state is None:
            state = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        
        # prepare elapsed_time broadcastable shape (batch,)
        if elapsed_time is None:
            dt = torch.ones(batch, device=device, dtype=dtype)
        else:
            if isinstance(elapsed_time, (float, int)):
                dt = torch.full((batch,), float(elapsed_time), device=device, dtype=dtype)
            else:
                dt = torch.as_tensor(elapsed_time, device=device, dtype=dtype).squeeze()
                if dt.dim() == 0:
                    dt = dt.repeat(batch)
                if dt.shape[0] != batch:
                    dt = dt.expand(batch)
        
        combined = torch.cat([input, state], dim=-1)
        feat = self.backbone(combined) if len(self.backbone) > 0 else combined
        g = self.head_g(feat)
        f = self.head_f(feat)
        h = self.head_h(feat)
        
        tau = self.tau().to(device=device, dtype=dtype)
        tau = torch.clamp(tau, min=_eps)
        
        dt_expanded = dt.unsqueeze(-1)

        #decay = torch.exp(- dt_expanded / tau)
        
        #nominal = decay * state + (1.0 - decay) * f

        # Compute time-gate s = sigmoid(- f * dt)
        s = torch.sigmoid(- f * dt_expanded)       # (batch, hidden)

        # Compute CfC update (Eq.10)
        new_state = s * g + (1.0 - s) * h         # (batch, hidden)
        '''
        if self.mode == "default":
            gate = torch.sigmoid(g)
            new_state = gate * h + (1.0 - gate) * nominal
        elif self.mode == "no_gate":
            new_state = nominal
        elif self.mode == "pure":
            new_state = nominal
        else:
            raise ValueError("Unknown mode")
        '''
        output = self.readout(new_state) if self.readout is not None else new_state
        
        if return_output:
            return output, new_state
        else:
            return new_state, new_state


class CfCLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        dropout: float = 0.0,
        **cell_kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_in = input_size if i == 0 else hidden_size
            self.cells.append(CfCCell(layer_in, hidden_size, **cell_kwargs))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed_time: Optional[torch.Tensor] = None,
        return_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_seq = input.dim() == 3
        if not is_seq:
            input = input.unsqueeze(1)
        
        batch, seq_len, _ = input.shape
        
        if state is None:
            state = torch.zeros(self.num_layers, batch, self.hidden_size, device=input.device, dtype=input.dtype)
        
        outputs = []
        if elapsed_time is None:
            dt_per_step = None
        else:
            dt_tensor = torch.as_tensor(elapsed_time, device=input.device, dtype=input.dtype)
            if dt_tensor.dim() == 0:
                dt_per_step = None
            elif dt_tensor.dim() == 1 and dt_tensor.shape[0] == batch:
                dt_per_step = dt_tensor
            elif dt_tensor.dim() == 2 and dt_tensor.shape == (batch, seq_len):
                dt_per_step = dt_tensor
            else:
                dt_per_step = dt_tensor
        
        for t in range(seq_len):
            x = input[:, t]
            dt_t = None
            if dt_per_step is None:
                dt_t = elapsed_time
            else:
                if dt_per_step.dim() == 2:
                    dt_t = dt_per_step[:, t]
                elif dt_per_step.dim() == 1:
                    dt_t = dt_per_step
                else:
                    dt_t = elapsed_time
            new_states = []
            for li, cell in enumerate(self.cells):
                s = state[li]
                out, new_s = cell(x, s, elapsed_time=dt_t, return_output=True)
                new_states.append(new_s)
                x = out
                if self.dropout is not None and li < self.num_layers - 1:
                    x = self.dropout(x)
            outputs.append(x)
            state = torch.stack(new_states, dim=0)
        
        out_seq = torch.stack(outputs, dim=1)
        if not is_seq:
            out_seq = out_seq.squeeze(1)
        
        if return_output:
            return out_seq, state
        else:
            return state, state

