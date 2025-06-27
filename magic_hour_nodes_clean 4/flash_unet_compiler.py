
class MagicHourFlashUNetCompiler:
    """torch.compile() wrapper (graceful fallback)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "compile"
    CATEGORY = "magic hour/model"
    def compile(self, model):
        try:
            import torch
            return (torch.compile(model),)
        except Exception as e:
            print("[Flash‑UNet] compile unavailable ->", e)
            return (model,)
