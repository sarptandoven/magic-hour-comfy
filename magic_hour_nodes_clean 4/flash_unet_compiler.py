"""
Magic-Hour Flash-UNet Compiler
==============================
• Transfers the incoming UNet to GPU *before* the sampler touches it.
• Then compiles it (torch.compile or CUDA-Graph fallback) and caches the result.
"""

from __future__ import annotations
import hashlib, inspect, sys, torch
from torch import nn
from typing import Dict

# ── in-process cache ──────────────────────────────────────────────────────────
_COMPILED_CACHE: Dict[str, nn.Module] = {}

# ── helpers ───────────────────────────────────────────────────────────────────
def _sha(unet: nn.Module) -> str:
    h = hashlib.sha256()
    for p in unet.state_dict().values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def _dummy_args(unet: nn.Module):
    sig = inspect.signature(unet.forward)
    latent = torch.randn(1, 4, 64, 64, device="cuda")
    zero   = torch.tensor(0.0, device="cuda")
    buf = []
    for name in sig.parameters:
        buf.append(latent if name in {"x", "latent"} else zero)
    return tuple(buf)

def _torch_compile(unet: nn.Module) -> nn.Module:
    return torch.compile(unet, mode="default", fullgraph=True)

def _cuda_graph(unet: nn.Module) -> nn.Module:
    unet.eval()
    args = _dummy_args(unet)
    for _ in range(3):                       # warm-up
        unet(*args)

    static_in  = [a.clone() for a in args]
    static_out = [torch.empty_like(o) for o in unet(*args)]

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = unet(*static_in)
        if isinstance(outs, (tuple, list)):
            for dst, src in zip(static_out, outs):
                dst.copy_(src)
        else:
            static_out[0].copy_(outs)

    class _Wrap(nn.Module):
        def __init__(self, g, sin, sout):
            super().__init__(); self.g, self.sin, self.sout = g, sin, sout
        def forward(self, *a):
            for buf, new in zip(self.sin, a): buf.copy_(new)
            self.g.replay()
            return tuple(self.sout) if len(self.sout) > 1 else self.sout[0]
    return _Wrap(g, static_in, static_out)

# ── ComfyUI node ──────────────────────────────────────────────────────────────
class MagicHourFlashUNetCompiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "unet": ("UNET",),
            "force_recompile": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("UNET",)
    RETURN_NAMES = ("unet",)
    FUNCTION     = "run"
    CATEGORY     = "magic hour/model"

    def run(self, unet: nn.Module, force_recompile: bool = False):
        # --- PRE-LOAD on GPU *before* anything else touches it -----------
        unet = unet.to(device="cuda", non_blocking=True)

        key = _sha(unet)
        if force_recompile and key in _COMPILED_CACHE:
            del _COMPILED_CACHE[key]

        if key in _COMPILED_CACHE:
            return (_COMPILED_CACHE[key],)

        try:
            compiled = _torch_compile(unet)
            print("[Flash-UNet] compiled via torch.compile")
        except Exception as e:
            print("[Flash-UNet] torch.compile failed:", e, file=sys.stderr)
            compiled = _cuda_graph(unet)
            print("[Flash-UNet] using CUDA-Graph fallback")

        _COMPILED_CACHE[key] = compiled
        return (compiled,)

# ── registration dicts ────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {"MagicHourFlashUNetCompiler": MagicHourFlashUNetCompiler}
NODE_DISPLAY_NAME_MAPPINGS = {"MagicHourFlashUNetCompiler": "MagicHour Flash-UNet Compiler"}
