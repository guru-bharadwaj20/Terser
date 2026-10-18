"""TextCNN span scorer (plan rows **8.2.1**, **8.2.2**, **8.2.3**).

Input: token embeddings ``(B, L, 384)`` (bge-small token states, computed upstream, so the
embedding table is not part of this model) and a padding mask ``(B, L)``.

* Two parallel ``Conv1d(384 -> 128)`` branches, kernel 3 and kernel 5, 'same' padding, ReLU.
* Concatenated per-token features ``(B, L, 256)``.
* **Token head** ``Linear(256 -> 1)``: per-token KEEP logit (span importance).
* **Prompt head** on a pooled 256-d vector (global max, global average, or both concatenated
  into 512-d): prompt-compressibility logit.

Parameter count (row 8.2.2; a conv layer has ``C_in * k * C_out`` weights and ``C_out``
biases, a linear layer ``m * k + k``):

=================  ==========================  =========
Layer              Weights + biases            Params
=================  ==========================  =========
Conv k=3           384 x 3 x 128 + 128         147,584
Conv k=5           384 x 5 x 128 + 128         245,888
Token head         256 x 1 + 1                 257
Prompt head        256 x 1 + 1                 257
**Total**                                      **393,986**
=================  ==========================  =========

With ``pooling="concat"`` the prompt head is ``512 + 1 = 513`` and the total 394,242.
:func:`textcnn_param_count` is the closed form, tested against ``sum(p.numel())``. Derivation:
``docs/derivations/deep_learning_intro.md`` section 4. ``torch`` is imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from torch import nn

__all__ = ["TextCNNConfig", "build_textcnn", "textcnn_param_count"]

POOLINGS: Final[tuple[str, ...]] = ("max", "avg", "concat")


class TextCNNConfig:
    """Architecture hyperparameters (defaults are the plan's)."""

    def __init__(
        self,
        embed_dim: int = 384,
        kernels: Sequence[int] = (3, 5),
        filters: int = 128,
        pooling: str = "max",
        dropout: float = 0.0,
    ) -> None:
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        if any(k % 2 == 0 or k < 1 for k in kernels):
            raise ValueError("'same' padding needs odd kernels")
        self.embed_dim = embed_dim
        self.kernels = tuple(kernels)
        self.filters = filters
        self.pooling = pooling
        self.dropout = dropout

    @property
    def feature_dim(self) -> int:
        return self.filters * len(self.kernels)

    @property
    def prompt_head_in(self) -> int:
        return 2 * self.feature_dim if self.pooling == "concat" else self.feature_dim


def textcnn_param_count(cfg: TextCNNConfig | None = None) -> int:
    """Closed-form parameter count: conv branches + token head + prompt head."""
    c = cfg or TextCNNConfig()
    convs = sum(c.embed_dim * k * c.filters + c.filters for k in c.kernels)
    return convs + (c.feature_dim + 1) + (c.prompt_head_in + 1)


def build_textcnn(cfg: TextCNNConfig | None = None) -> nn.Module:
    """The TextCNN as an ``nn.Module``; ``forward(x, mask)`` returns (token_logits, prompt_logit).

    ``x``: (B, L, embed_dim) float; ``mask``: (B, L) bool, True for real tokens (optional).
    Padded positions are excluded from both poolings.
    """
    import torch
    from torch import nn

    c = cfg or TextCNNConfig()

    class TextCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = c
            self.convs = nn.ModuleList(
                nn.Conv1d(c.embed_dim, c.filters, k, padding=(k - 1) // 2) for k in c.kernels
            )
            self.dropout = nn.Dropout(c.dropout)
            self.token_head = nn.Linear(c.feature_dim, 1)
            self.prompt_head = nn.Linear(c.prompt_head_in, 1)

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Any:
            if mask is None:
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            h = x.transpose(1, 2)  # (B, D, L)
            feats = torch.cat([torch.relu(conv(h)) for conv in self.convs], dim=1)
            feats = self.dropout(feats.transpose(1, 2))  # (B, L, F)
            token_logits = self.token_head(feats).squeeze(-1)
            m = mask.unsqueeze(-1)
            pooled_max = feats.masked_fill(~m, float("-inf")).amax(dim=1)
            pooled_avg = (feats * m).sum(1) / m.sum(1).clamp(min=1)
            if c.pooling == "max":
                pooled = pooled_max
            elif c.pooling == "avg":
                pooled = pooled_avg
            else:
                pooled = torch.cat([pooled_max, pooled_avg], dim=-1)
            return token_logits, self.prompt_head(pooled).squeeze(-1)

    return TextCNN()
