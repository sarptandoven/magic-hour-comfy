"""Magic Hour Node Pack (clean)"""
from .flash_unet_compiler import MagicHourFlashUNetCompiler
from .smart_quant_cascade import MagicHourSmartQuantCascade
from .sampler_custom_advanced import MagicHourSamplerCustomAdvancedEarlyExit

NODE_CLASS_MAPPINGS = {
    "MagicHourFlashUNetCompiler": MagicHourFlashUNetCompiler,
    "MagicHourSmartQuantCascade": MagicHourSmartQuantCascade,
    "MagicHourSamplerCustomAdvancedEarlyExit": MagicHourSamplerCustomAdvancedEarlyExit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicHourFlashUNetCompiler": "MagicHour Flash‑UNet Compiler",
    "MagicHourSmartQuantCascade": "MagicHour Smart‑Quant Cascade",
    "MagicHourSamplerCustomAdvancedEarlyExit": "MagicHour Sampler Advanced (Random Early Exit)",
}
