import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, dim - 1)

    def forward(self, timesteps, device):
        t = (torch.arange(timesteps, dtype=torch.float32, device=device) / timesteps).unsqueeze(-1)
        return torch.cat([self.lin(t), torch.sin(self.per(t))], dim=-1)


class TwoWayBlock(nn.Module):
    def __init__(self, dim, nhead, feedforward_dim, dropout):
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cross_sec = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, stock_mask=None):
        squeeze_batch = x.dim() == 3
        if squeeze_batch:
            x = x.unsqueeze(0)
            if stock_mask is not None:
                stock_mask = stock_mask.unsqueeze(0)

        batch_size, num_stocks, timesteps, dim = x.shape

        x = x.reshape(batch_size * num_stocks, timesteps, dim)
        x = self.temporal(x)
        x = x.reshape(batch_size, num_stocks, timesteps, dim)

        x = x.permute(0, 2, 1, 3).reshape(batch_size * timesteps, num_stocks, dim)
        if stock_mask is not None:
            pad_mask = (stock_mask == 0).unsqueeze(1).expand(-1, timesteps, -1).reshape(batch_size * timesteps, num_stocks)
        else:
            pad_mask = None
        x = self.cross_sec(x, src_key_padding_mask=pad_mask)
        x = x.reshape(batch_size, timesteps, num_stocks, dim).permute(0, 2, 1, 3)

        if squeeze_batch:
            return x.squeeze(0)
        return x


class QuantV4(nn.Module):
    def __init__(self, nf, d=128, nh=4, nl=3, ff=256, drop=0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(nf, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.t2v = Time2Vec(d)
        self.blocks = nn.ModuleList([TwoWayBlock(d, nh, ff, drop) for _ in range(nl)])
        self.pool = nn.Sequential(nn.Linear(d, d // 4), nn.Tanh(), nn.Linear(d // 4, 1))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d // 2, 1))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, stock_mask=None):
        squeeze_batch = x.dim() == 3
        if squeeze_batch:
            x = x.unsqueeze(0)
            if stock_mask is not None:
                stock_mask = stock_mask.unsqueeze(0)

        _, _, timesteps, _ = x.shape
        x = self.proj(x) + self.t2v(timesteps, x.device).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, stock_mask)
        weights = torch.softmax(self.pool(x), dim=2)
        x = (x * weights).sum(dim=2)
        x = self.head(x).squeeze(-1)

        if squeeze_batch:
            return x.squeeze(0)
        return x
