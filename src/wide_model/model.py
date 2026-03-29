from pathlib import Path
import torch
import torch.nn as nn


class _LNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LNResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class WideEncoderDecoder(nn.Module):
    def __init__(
        self,
        in_features: int      = 130,
        width: int            = 2048,
        n_decoder_blocks: int = 5,
        out_features: int     = 512,
        dropout: float        = 0.1,
    ):
        super().__init__()
        self.in_features      = in_features
        self.width            = width
        self.n_decoder_blocks = n_decoder_blocks
        self.out_features     = out_features
        self.dropout          = dropout

        self.encoder = _LNBlock(in_features, width, dropout)
        self.decoder = nn.Sequential(
            *[_LNResBlock(width, dropout) for _ in range(n_decoder_blocks)]
        )
        self.output = nn.Linear(width, out_features)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return self.output(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict":        self.state_dict(),
            "in_features":       self.in_features,
            "width":             self.width,
            "n_decoder_blocks":  self.n_decoder_blocks,
            "out_features":      self.out_features,
            "dropout":           self.dropout,
        }, path)

    def load(path: Path) -> "WideEncoderDecoder":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = WideEncoderDecoder(
            in_features      = ckpt["in_features"],
            width            = ckpt["width"],
            n_decoder_blocks = ckpt["n_decoder_blocks"],
            out_features     = ckpt["out_features"],
            dropout          = ckpt["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        return model
