from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def get_device() -> str:
    return str(DEVICE)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CHECKPOINTS = {
    "zero-shot": MODELS_DIR / "RealESRGAN_x4plus.pth",
    "head-only": MODELS_DIR / "esrgan_head_only.pth",
    "partial": MODELS_DIR / "esrgan_partial.pth",
    "lora": MODELS_DIR / "esrgan_lora.pth",
    "full": MODELS_DIR / "esrgan_full.pth",
}

VALID_STRATEGIES = set(CHECKPOINTS.keys())


# ============================================================
# MODEL CACHE
# ============================================================

_CURRENT_MODEL: nn.Module | None = None
_CURRENT_STRATEGY: str | None = None


def unload_current_model() -> None:
    global _CURRENT_MODEL
    global _CURRENT_STRATEGY

    if _CURRENT_MODEL is not None:
        try:
            _CURRENT_MODEL.cpu()
        except Exception:
            pass

        del _CURRENT_MODEL

    _CURRENT_MODEL = None
    _CURRENT_STRATEGY = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================
# RESIDUAL DENSE BLOCK
# ============================================================

class ResidualDenseBlock(nn.Module):
    def __init__(self,nf: int = 64,gc: int = 32,) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf,gc,kernel_size=3,stride=1,padding=1,)
        self.conv2 = nn.Conv2d(nf + gc,gc,kernel_size=3,stride=1,padding=1,)
        self.conv3 = nn.Conv2d(nf + gc * 2,gc,kernel_size=3,stride=1,padding=1,)
        self.conv4 = nn.Conv2d(nf + gc * 3,gc,kernel_size=3,stride=1,padding=1,)
        self.conv5 = nn.Conv2d(nf + gc * 4,nf,kernel_size=3,stride=1,padding=1,)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True,)

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1],dim=1,)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2],dim=1,)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3],dim=1,)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4],dim=1,))
        return x + 0.2 * x5


# ============================================================
# RRDB
# ============================================================

class RRDB(nn.Module):
    def __init__(self,nf: int = 64,gc: int = 32,) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf=nf,gc=gc,)
        self.rdb2 = ResidualDenseBlock(nf=nf,gc=gc,)
        self.rdb3 = ResidualDenseBlock(nf=nf,gc=gc,)

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out



class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch,num_feat,kernel_size=3,stride=1,padding=1,)
        self.body = nn.Sequential(
            *[
                RRDB(
                    nf=num_feat,
                    gc=num_grow_ch,
                )
                for _ in range(num_block)
            ]
        )
        self.conv_body = nn.Conv2d(num_feat,num_feat,kernel_size=3,stride=1,padding=1,)
        self.conv_up1 = nn.Conv2d(num_feat,num_feat,kernel_size=3,stride=1,padding=1,)
        self.conv_up2 = nn.Conv2d(num_feat,num_feat,kernel_size=3,stride=1,padding=1,)
        self.conv_hr = nn.Conv2d(num_feat,num_feat,kernel_size=3,stride=1,padding=1,)
        self.conv_last = nn.Conv2d(num_feat,num_out_ch,kernel_size=3,stride=1,padding=1,)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True,)

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(
            self.conv_up1(
                F.interpolate(
                    feat,
                    scale_factor=2,
                    mode="nearest",
                )
            )
        )
        feat = self.lrelu(
            self.conv_up2(
                F.interpolate(
                    feat,
                    scale_factor=2,
                    mode="nearest",
                )
            )
        )
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        return out



class LoRAConv2d(nn.Module):
    def __init__(self,conv: nn.Conv2d,rank: int = 8,alpha: float = 8.0,) -> None:
        super().__init__()
        self.base_conv = conv
        self.rank = rank
        self.alpha = alpha

        for p in self.base_conv.parameters():
            p.requires_grad_(False)

        # IMPORTANT:
        # Actual checkpoint has (8, 64, 3, 3)
        self.lora_down = nn.Conv2d(conv.in_channels,rank,kernel_size=3,stride=1,padding=1,bias=False,)

        # Actual checkpoint has (out_channels, 8, 1, 1)
        self.lora_up = nn.Conv2d(rank,conv.out_channels,kernel_size=1,stride=1,padding=0,bias=False,)

        # Same initialization family used for the training code.
        nn.init.kaiming_uniform_(self.lora_down.weight,a=np.sqrt(5),)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        base = self.base_conv(x)
        update = (
            self.lora_up(
                self.lora_down(x)
            )
            * (
                self.alpha / self.rank
            )
        )
        return base + update



def apply_lora(model: RRDBNet,) -> RRDBNet:
    target_layers = [
        "conv_body",
        "conv_up1",
        "conv_up2",
        "conv_hr",
        "conv_last",
    ]
    for name in target_layers:
        old = getattr(model,name,)
        if not isinstance(old,nn.Conv2d,):
            raise RuntimeError(
                f"Expected {name} to be nn.Conv2d, "
                f"got {type(old)}"
            )

        setattr(
            model,
            name,
            LoRAConv2d(
                old,
                rank=8,
                alpha=8,
            ),
        )

    # Match training strategy:
    # freeze everything, then LoRA tensors only.
    for p in model.parameters():
        p.requires_grad_(False)

    for module in model.modules():

        if isinstance(
            module,
            LoRAConv2d,
        ):

            module.lora_down.weight.requires_grad_(True)
            module.lora_up.weight.requires_grad_(True)

    return model



def build_base_model() -> RRDBNet:
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )


# ============================================================
# CHECKPOINT EXTRACTION
# ============================================================

def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    # Support wrappers, although your uploaded LoRA checkpoint
    # is already an OrderedDict state_dict.
    for key in (
        "params_ema",
        "params",
        "state_dict",
        "model_state_dict",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate,dict,):
            checkpoint = candidate
            break

    result: dict[str, torch.Tensor] = {}

    for key, value in checkpoint.items():

        if not torch.is_tensor(value):
            continue
        clean_key = key
        while clean_key.startswith("module."):
            clean_key = clean_key[7:]
        result[clean_key] = value
    if not result:
        raise RuntimeError(
            "No tensor weights were found in checkpoint."
        )
    return result


# ============================================================
# LORA CHECKPOINT DIAGNOSTICS
# ============================================================

def diagnose_lora(state: dict[str, torch.Tensor],) -> dict[str, Any]:
    down_keys = sorted(k
        for k in state
        if ".lora_down.weight" in k
    )
    up_keys = sorted(
        k
        for k in state
        if ".lora_up.weight" in k
    )
    print()
    print("[LoRA] Number of lora_down tensors:",len(down_keys),)
    print("[LoRA] Number of lora_up tensors:",len(up_keys),)
    down_abs_sum = 0.0
    up_abs_sum = 0.0
    for key in down_keys:

        tensor = state[key].float()

        print(f"[LoRA] DOWN {key}: "f"shape={tuple(tensor.shape)} "f"abs_mean={tensor.abs().mean().item():.8f} "f"abs_max={tensor.abs().max().item():.8f}")

        down_abs_sum += float(tensor.abs().sum().item())
        
    for key in up_keys:

        tensor = state[key].float()
        print(f"[LoRA] UP   {key}: "f"shape={tuple(tensor.shape)} "f"abs_mean={tensor.abs().mean().item():.8f} "f"abs_max={tensor.abs().max().item():.8f}")

        up_abs_sum += float(tensor.abs().sum().item())

    print("[LoRA] Total DOWN absolute sum:",down_abs_sum,)
    print("[LoRA] Total UP absolute sum:",up_abs_sum,)

    if up_abs_sum == 0.0:
        print()
        print("[LoRA WARNING] lora_up is entirely zero.")
        print("[LoRA WARNING] This would make the model ""behave like the base model.")
        print()

    else:
        print("[LoRA] Learned update detected.")

    return {
        "down_count": len(down_keys),
        "up_count": len(up_keys),
        "down_abs_sum": down_abs_sum,
        "up_abs_sum": up_abs_sum,
    }


# ============================================================
# STRICT COMPLETE CHECKPOINT LOAD
# ============================================================

def load_complete_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    strategy: str,
) -> None:

    print()
    print("=" * 70)
    print(f"[ESRGAN] Loading {strategy}: "f"{checkpoint_path.name}")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path,map_location="cpu",)

    state = extract_state_dict(checkpoint)

    del checkpoint

    gc.collect()

    model_state = model.state_dict()

    model_keys = set(model_state.keys())

    checkpoint_keys = set(state.keys())

    missing = sorted(model_keys - checkpoint_keys)

    unexpected = sorted(checkpoint_keys - model_keys)

    matched = (model_keys & checkpoint_keys)

    print("[ESRGAN] Model tensor count:",len(model_keys),)
    print("[ESRGAN] Checkpoint tensor count:",len(checkpoint_keys),)
    print("[ESRGAN] Exact matching tensors:",len(matched),)

    # --------------------------------------------------------
    # LoRA-specific inspection
    # --------------------------------------------------------

    if strategy == "lora":
        diagnose_lora(state)

    # --------------------------------------------------------
    # Show architecture mismatch
    # --------------------------------------------------------

    if missing:
        print()
        print("[ESRGAN] MISSING KEYS:")

        for key in missing[:50]:
            print("   ",key,)

    if unexpected:
        print()
        print("[ESRGAN] UNEXPECTED KEYS:")

        for key in unexpected[:50]:
            print("   ",key,)

    if missing or unexpected:

        raise RuntimeError("\n""Checkpoint architecture mismatch.\n"
            f"Missing keys: {len(missing)}\n"
            f"Unexpected keys: {len(unexpected)}"
        )

    # --------------------------------------------------------
    # STRICT COMPLETE LOAD
    # --------------------------------------------------------

    result = model.load_state_dict(
        state,
        strict=True,
    )

    print()
    print("[ESRGAN] load_state_dict result:",result,)
    print("[ESRGAN] COMPLETE STRICT LOAD SUCCESS.")
    print("=" * 70)
    print()

    del state
    del model_state
    gc.collect()


# ============================================================
# LOAD STRATEGY MODEL
# ============================================================

def load_strategy_model(
    strategy: str,
) -> RRDBNet:

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    checkpoint_path = CHECKPOINTS[
        strategy
    ]

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Build base model
    # --------------------------------------------------------

    model = build_base_model()

    if strategy == "lora":

        print(
            "[ESRGAN] Applying EXACT checkpoint LoRA architecture..."
        )

        model = apply_lora(
            model
        )

    load_complete_checkpoint(
        model,
        checkpoint_path,
        strategy,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================
# GET MODEL
# ============================================================

def get_model(
    strategy: str,
) -> nn.Module:

    global _CURRENT_MODEL
    global _CURRENT_STRATEGY

    if (
        _CURRENT_MODEL is not None
        and _CURRENT_STRATEGY == strategy
    ):

        return _CURRENT_MODEL

    unload_current_model()

    print(f"[ESRGAN] Preparing strategy: {strategy}")

    model = load_strategy_model(strategy)
    model = model.to(DEVICE)
    model.eval()

    _CURRENT_MODEL = model
    _CURRENT_STRATEGY = strategy

    if torch.cuda.is_available():
        print("[ESRGAN] GPU:",torch.cuda.get_device_name(0),)
        print("[ESRGAN] Allocated VRAM:",round(torch.cuda.memory_allocated()/ 1024**2,2,),"MB",)
        print("[ESRGAN] Reserved VRAM:",round(torch.cuda.memory_reserved()/ 1024**2,2,),"MB",)
    return model


# ============================================================
# IMAGE -> TENSOR
# ============================================================

def image_to_tensor(
    image: Image.Image | np.ndarray,
) -> torch.Tensor:
    if isinstance(image,Image.Image,):
        image = image.convert("RGB")
        array = np.asarray(
            image,
            dtype=np.float32,
        )
        array /= 255.0

    elif isinstance(image,np.ndarray,):
        array = image.astype(
            np.float32,
            copy=False,
        )
        if array.ndim == 2:
            array = np.stack(
                [array] * 3,
                axis=-1,
            )
        if array.ndim != 3:
            raise ValueError(
                "Expected HxWxC image."
            )

        if array.shape[-1] == 4:
            array = array[..., :3]

        if array.shape[-1] != 3:
            raise ValueError(
                "Expected RGB image."
            )
        if array.max() > 1.0:
            array /= 255.0

    else:
        raise TypeError(
            "Expected PIL.Image or numpy.ndarray."
        )
    array = np.clip(
        array,
        0.0,
        1.0,
    )
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2,0,1,)
    tensor = tensor.unsqueeze(0)
    
    return tensor.to(
        DEVICE,
        dtype=torch.float32,
    )


# ============================================================
# TENSOR -> IMAGE
# ============================================================

def tensor_to_image(tensor: torch.Tensor,) -> Image.Image:
    tensor = tensor.detach()
    tensor = tensor.clamp(0.0,1.0,)
    tensor = tensor.cpu()
    array = tensor[0].permute(1,2,0,).numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(array,mode="RGB",)

# ============================================================
# INFERENCE
# ============================================================

def upscale_image(image: Image.Image | np.ndarray,strategy: str = "zero-shot",) -> Image.Image:

    model = get_model(strategy)
    x = None
    y = None
    try:
        x = image_to_tensor(image)
        with torch.inference_mode():
            y = model(x)
            result = tensor_to_image(y)
        return result
    except torch.cuda.OutOfMemoryError as exc:
        unload_current_model()

        raise RuntimeError(
            "CUDA out of memory during ESRGAN inference. "
            "Try a smaller input image."
        ) from exc

    finally:
        if x is not None:
            del x
        if y is not None:
            del y
        unload_current_model()


# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================

def infer_esrgan(image: Image.Image | np.ndarray,strategy: str = "zero-shot",) -> Image.Image:
    return upscale_image(image,strategy,)

def load_model(strategy: str = "zero-shot",) -> nn.Module:
    return get_model(strategy)

# ============================================================
# RUNTIME INFO
# ============================================================

def get_runtime_info() -> dict[str, Any]:

    info: dict[str, Any] = {
        "device": str(DEVICE),
        "cuda_available":
            torch.cuda.is_available(),
        "model_loaded":
            _CURRENT_MODEL is not None,
        "loaded_strategy":
            _CURRENT_STRATEGY,
    }

    if torch.cuda.is_available():

        info["gpu"] = (
            torch.cuda.get_device_name(0)
        )

        info["allocated_mb"] = round(
            torch.cuda.memory_allocated()
            / 1024**2,
            2,
        )

        info["reserved_mb"] = round(
            torch.cuda.memory_reserved()
            / 1024**2,
            2,
        )

    return info