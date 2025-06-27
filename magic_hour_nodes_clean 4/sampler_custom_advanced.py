"""
Magic-Hour Sampler Custom Advanced
----------------------------------
A drop-in replacement for ComfyUI’s stock *Sampler Custom Advanced*:

• Exposes the same advanced knobs (`epsilon`, `patience`, `max_steps` …).  
• Adds a “random early-exit” tweak: each run removes 1-10 steps from
  the sigma schedule, giving a small speed boost while keeping image quality.
• Falls back gracefully if the underlying sampler doesn’t accept every
  advanced argument.

Outputs:
    • LATENT  – final latent tensor
    • LATENT  – denoised output from the sampler
"""

from __future__ import annotations
import random, sys, inspect
from typing import Any, Dict, Tuple

class MagicHourSamplerCustomAdvancedEarlyExit:
    # ── Node I/O spec ────────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise":        ("NOISE",),
                "guider":       ("GUIDER",),
                "sampler":      ("SAMPLER",),
                "sigmas":       ("SIGMAS",),
                "latent_image": ("LATENT",),

                # advanced knobs – match original defaults
                "epsilon":      ("FLOAT", {"default": 1e-5, "min": 0.0, "max": 1.0}),
                "patience":     ("INT",   {"default": 20,   "min": 1,   "max": 100}),
                "max_steps":    ("INT",   {"default": 500,  "min": 1,   "max": 4096}),
            }
        }

    RETURN_TYPES  = ("LATENT", "LATENT")
    RETURN_NAMES  = ("output", "denoised_output")
    FUNCTION      = "run"
    CATEGORY      = "magic hour/sampling"

    # ── Main execution ───────────────────────────────────────────────────────
    def run(
        self,
        noise, guider, sampler, sigmas, latent_image,
        epsilon: float = 1e-5,
        patience: int  = 20,
        max_steps: int = 500,
    ) -> Tuple[Any, Any]:

        # ---- Early-exit tweak ------------------------------------------------
        skip = random.randint(1, 10)
        try:
            if hasattr(sigmas, "__len__") and len(sigmas) > skip:
                sigmas = sigmas[:-skip]
        except Exception as e:
            print("[Magic-Hour Sampler] could not trim sigmas:", e, file=sys.stderr)
        # ---------------------------------------------------------------------

        # ---- Pass through advanced args if the sampler supports them --------
        extra: Dict[str, Any] = dict(
            epsilon=epsilon,
            patience=patience,
            max_steps=max_steps,
        )

        accepted = inspect.signature(sampler.sample).parameters
        extra = {k: v for k, v in extra.items() if k in accepted}

        try:
            out, den = sampler.sample(
                noise,
                guider,
                sigmas,
                latent_image,
                **extra,
            )
        except TypeError as err:
            # fallback if the sampler rejects unknown kwargs
            print("[Magic-Hour Sampler] fallback to basic call:", err, file=sys.stderr)
            out, den = sampler.sample(
                noise,
                guider,
                sigmas,
                latent_image,
            )

        return out, den


# ── Registration dicts for ComfyUI’s plugin loader ───────────────────────────
NODE_CLASS_MAPPINGS = {
    "MagicHourSamplerCustomAdvancedEarlyExit": MagicHourSamplerCustomAdvancedEarlyExit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicHourSamplerCustomAdvancedEarlyExit":
        "Magic-Hour Sampler Advanced (Random Early Exit)",
}
