import os
import warnings

import torch
import torch.nn as nn

from src.components.encoders.feature_encoder import PositionalEncoding


class NavigationPolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        use_transformer=False,
        use_map=True,
        value_head=False,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_len=256,
        device=None,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_map = use_map

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_transformer:
            self.pos_encoder = PositionalEncoding(input_dim, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
            )
            self.core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.core_output_dim = input_dim
            self._expected_input_dim = input_dim
        else:
            self.core = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.core_output_dim = hidden_dim
            self._expected_input_dim = getattr(self.core, "input_size", input_dim)

        self.shared = nn.Sequential(
            nn.Linear(self.core_output_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

        self.value_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)).to(self.device) if value_head else None
        )

    def forward(self, seq, hidden=None, pad_mask=None):
        """
        seq: Tensor [B, T, D]
        hidden: (h, c) for LSTM, None for Transformer
        pad_mask: Optional [B, T] bool mask for Transformer (True=pad)
        """
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)

        if not self.use_transformer:
            expected = getattr(self.core, "input_size", self._expected_input_dim)
        else:
            expected = self._expected_input_dim

        current = seq.size(-1)
        if current != expected:
            if current < expected:
                pad_size = expected - current
                warnings.warn(
                    f"[NavigationPolicy] Input feature dim ({current}) smaller than expected ({expected}); padding zeros.",
                    RuntimeWarning,
                )
                pad_shape = list(seq.shape[:-1]) + [pad_size]
                pad = seq.new_zeros(pad_shape)
                seq = torch.cat([seq, pad], dim=-1)
            else:
                warnings.warn(
                    f"[NavigationPolicy] Input feature dim ({current}) larger than expected ({expected}); truncating features.",
                    RuntimeWarning,
                )
                seq = seq[..., :expected]

        if self.use_transformer:
            hidden = None
            seq = self.pos_encoder(seq)
            # Transformer expects src_key_padding_mask (True = PAD)
            out = self.core(seq, src_key_padding_mask=pad_mask)
        else:
            out, hidden = self.core(seq, hidden)

        out = self.shared(out)
        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1) if self.value_head is not None else None
        return logits, value, hidden

    def save_model(self, path):
        """
        Saves the model parameters and configs.
        """
        os.makedirs(path, exist_ok=True)
        output_dim = (self.policy_head[-1].out_features
                      if hasattr(self, "policy_head") and len(self.policy_head) > 0
                      else getattr(self, "output_dim", None))
        filename = f"navigation_policy_{self.input_dim}_{self.hidden_dim}_{output_dim}_{self.use_transformer}_{self.use_map}.pth"
        payload = {
            "state_dict": self.state_dict(),
            "meta": {
                "version": "pol_v1",
                "input_dim": int(getattr(self, "input_dim", 0)),
                "hidden_dim": int(getattr(self, "hidden_dim", 0)),
                "output_dim": int(output_dim) if output_dim is not None else None,
                "use_transformer": bool(getattr(self, "use_transformer", False)),
                "use_map": bool(getattr(self, "use_map", True)),
                "value_head": self.value_head is not None,
            }
        }
        torch.save(payload, os.path.join(path, filename))

    def load_weights(self, model_path, device="cpu", strict = True):
        """
        Loads model weights into an existing NavigationPolicy instance.
        """
        payload = torch.load(model_path, map_location=device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.load_state_dict(state_dict, strict=strict)
        self.to(device)
        return payload.get("meta", {})
