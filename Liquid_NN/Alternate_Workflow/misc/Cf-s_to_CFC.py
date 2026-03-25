import torch 
import torch.nn as nn


class CfCCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden)
        )
        self.f_head = nn.Linear(shared_hidden, hidden_dim)
        self.g_head = nn.Linear(shared_hidden, hidden_dim)
        self.h_head = nn.Linear(shared_hidden, hidden_dim)

    def forward(self, x, u, dt):
        inp = torch.cat([x, u], dim=-1)
        feat = self.shared(inp)
        f = F.softplus(self.f_head(feat))
        g = self.g_head(feat)
        h = self.h_head(feat)
        gate = torch.sigmoid(- f * dt)
        x_next = gate * g + (1.0 - gate) * h
        return x_next
