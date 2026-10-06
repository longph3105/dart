"""Compatibility of the DARTF rewrites with every transformers release that ships SAM 3.

The exact rewrites (window-major token order, the RoPE fold and the RoPE GEMM plugin) need SAM 3's per-token rotary tables
as tensors of shape [tokens, head_dim]:
  - transformers <= 5.16 stores them as buffers `rotary_emb.rope_embeddings_cos / rope_embeddings_sin`;
  - transformers >= 5.17 computes them at run time from `inv_freq` and each layer's `position_ids`.
`ensure_rope_tables(vit)` gives every layer the table form on any version: the tables are taken from the model itself (buffers,
or the rotary module's own forward), cross-checked against an independent computation from the config, and installed as a
module with the historical buffer names, so the exported ONNX graph and every downstream tool are identical across versions.
A future change of the rotary semantics fails the cross-check instead of producing a silently wrong graph."""
import torch
import torch.nn as nn


class RopeTables(nn.Module):
    """Fixed rotary tables; accepts and ignores the call arguments of any transformers version."""

    def __init__(self, cos, sin):
        super().__init__()
        self.rope_embeddings_cos = nn.Buffer(cos, persistent=False)
        self.rope_embeddings_sin = nn.Buffer(sin, persistent=False)

    def forward(self, *args, **kwargs):
        return self.rope_embeddings_cos, self.rope_embeddings_sin


def _grid(cfg, layer):
    """(side, scale) of the layer's token grid: windowed layers use a window_size grid, global layers the full patch grid
    with positions scaled to window units (SAM 3's interpolated RoPE)."""
    ws = cfg.window_size
    if layer.window_size > 0:
        return layer.window_size, 1.0
    img = cfg.image_size if isinstance(cfg.image_size, int) else cfg.image_size[0]
    patch = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    side = img // patch
    return side, ws / side


def reference_tables(cfg, layer):
    """Axial 2D RoPE of SAM 3's ViT from first principles: [side*side, head_dim] cos/sin, x varies fastest, pairs interleaved."""
    dim = cfg.hidden_size // cfg.num_attention_heads
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        theta = cfg.rope_parameters["rope_theta"]
    side, scale = _grid(cfg, layer)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].double() / dim))
    idx = torch.arange(side * side)
    x = (idx % side).double() * scale
    y = torch.div(idx, side, rounding_mode="floor").double() * scale
    ang = torch.cat([torch.outer(x, freqs), torch.outer(y, freqs)], -1).repeat_interleave(2, -1)
    return ang.cos().float(), ang.sin().float()


@torch.no_grad()
def _model_tables(layer):
    re = layer.rotary_emb
    if hasattr(re, "rope_embeddings_cos"):                       # transformers <= 5.16, or already converted
        return re.rope_embeddings_cos, re.rope_embeddings_sin
    if hasattr(layer, "position_ids"):                           # transformers >= 5.17
        pos = layer.position_ids
        cos, sin = re(torch.zeros((), device=pos.device), pos)
        return cos.reshape(-1, cos.shape[-1]), sin.reshape(-1, sin.shape[-1])
    return None


@torch.no_grad()
def ensure_rope_tables(vit, atol=1e-5):
    """Install fixed [tokens, head_dim] rotary tables on every layer of an HF Sam3ViTModel (idempotent)."""
    for i, layer in enumerate(vit.layers):
        if isinstance(layer.rotary_emb, RopeTables):
            continue
        ref_cos, ref_sin = reference_tables(vit.config, layer)
        got = _model_tables(layer)
        if got is None:
            raise RuntimeError(f"layer {i}: unrecognized rotary embedding {type(layer.rotary_emb).__name__} in this transformers "
                               "version; please report it (DARTF needs SAM 3's per-token rotary tables)")
        cos, sin = (t.detach().float().cpu() for t in got)
        if cos.shape != ref_cos.shape or (cos - ref_cos).abs().max() > atol or (sin - ref_sin).abs().max() > atol:
            raise RuntimeError(f"layer {i}: the model's rotary tables differ from SAM 3's axial RoPE (shape {tuple(cos.shape)} vs "
                               f"{tuple(ref_cos.shape)}); this transformers version changed the rotary semantics, please report it")
        dev = got[0].device
        layer.rotary_emb = RopeTables(cos.to(dev), sin.to(dev))
    return vit
