
import copy, sys
from typing import Any
try:
    import torch, torch.ao.quantization as tq
    from torch import nn
except ModuleNotFoundError:
    torch, tq, nn = None, None, None

def _log(m): print("[Smart‑Quant]", m, file=sys.stderr)

def _unwrap(m):
    for attr in ("inner_model","model","base","_model"):
        inner = getattr(m, attr, None)
        if inner is not None:
            return _unwrap(inner)
    return m

def _int8_quant(fp: nn.Module):
    if torch is None or tq is None:
        return None
    try:
        qconfig = tq.get_default_qconfig("fbgemm")
        m = copy.deepcopy(fp).cpu().eval()
        m = tq.quantize_fx.convert_fx(tq.quantize_fx.prepare_fx(m,{"":qconfig}))
        return m.to(fp.device) if hasattr(fp,"device") else m
    except Exception as e:
        _log(f"INT8 convert failed -> {e}")
        return None

def _bnb_quant(fp: nn.Module, bits: int):
    """
    Replace every nn.Linear with bitsandbytes quantized equivalent
    without editing the module dictionary during iteration.
    """
    try:
        import bitsandbytes as bnb
    except ModuleNotFoundError:
        _log("bitsandbytes not installed")
        return None
    LinearMap = {8: bnb.nn.Linear8bitLt, 4: bnb.nn.Linear4bit}
    Lcls = LinearMap.get(bits)
    if Lcls is None:
        return None
    m = copy.deepcopy(fp).cpu().eval()
    replace_pairs = []
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear):
            replace_pairs.append((name, mod))
    for name, mod in replace_pairs:
        parent_path = name.split(".")[:-1]
        attr_name = name.split(".")[-1]
        parent = m
        for part in parent_path:
            parent = getattr(parent, part)
        repl = Lcls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        setattr(parent, attr_name, repl)
    return m.to(fp.device) if hasattr(fp, "device") else m


class _Cascade(nn.Module):
    def __init__(self, fp, q, n):
        super().__init__()
        self.fp,self.q,self.n = fp,q,int(n)
        self.register_buffer("_i", torch.zeros(1,dtype=torch.int) if torch else None)
    def forward(self,*a,**k):
        if torch:
            use=int(self._i.item())<self.n and self.q is not None
            self._i+=1
            return (self.q if use else self.fp)(*a,**k)
        return self.fp(*a,**k)

class MagicHourSmartQuantCascade:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "model":("MODEL",),
            "q_type":(["INT8","Q8","Q4"],),
            "steps_q":("INT",{"default":8,"min":1,"max":50})
        }}
    RETURN_TYPES=("MODEL",)
    FUNCTION="apply"
    CATEGORY="magic hour/model"
    def apply(self, model: Any, q_type="INT8", steps_q: int=8):
        if torch is None:
            _log("PyTorch unavailable -> FP path")
            return (model,)
        fp=_unwrap(model)
        if not isinstance(fp, nn.Module):
            _log("No nn.Module inside model -> FP path")
            return (model,)
        q=None
        if q_type=="INT8": q=_int8_quant(fp)
        else: q=_bnb_quant(fp, 8 if q_type=="Q8" else 4)
        if q is None:
            _log("Quant failed -> FP path")
            return (model,)
        cascade=_Cascade(fp,q,steps_q)
        if hasattr(model,"inner_model"):
            model.inner_model=cascade
            return (model,)
        return (cascade,)
