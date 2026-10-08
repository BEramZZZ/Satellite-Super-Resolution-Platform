from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as load_safetensors

try:
    from diffusers import StableDiffusionUpscalePipeline
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Diffusion dependencies are missing. Install: "
        "pip install diffusers transformers accelerate safetensors"
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

DIFFUSION_MODEL_ID = os.getenv(
    "DIFFUSION_MODEL_ID",
    "stabilityai/stable-diffusion-x4-upscaler",
)

# Interactive inference settings
DIFFUSION_STEPS = int(
    os.getenv("DIFFUSION_INFERENCE_STEPS", "30")
)

DIFFUSION_NOISE_LEVEL = int(
    os.getenv("DIFFUSION_NOISE_LEVEL", "20")
)

DIFFUSION_GUIDANCE_SCALE = float(
    os.getenv("DIFFUSION_GUIDANCE_SCALE", "9.0")
)

DIFFUSION_PROMPT = os.getenv(
    "DIFFUSION_PROMPT",
    "a high resolution aerial satellite photograph, sharp detail, realistic terrain",
)

DIFFUSION_SEED_BASE = int(
    os.getenv("DIFFUSION_SEED_BASE", "1234")
)

# RTX 2060 6 GB:
# keep the pipeline CPU-offloaded so the entire model does not stay in VRAM.
DIFFUSION_CPU_OFFLOAD = (
    os.getenv("DIFFUSION_CPU_OFFLOAD", "1") == "1"
)


# =============================================================================
# Local model locations
# =============================================================================

# Default:
#
# backend/
#   checkpoints/
#     diffusion/
#       head_only/
#       partial/
#       lora/
#       full/
#
# Each strategy directory is expected to contain its Diffusers checkpoint.
#
# Example:
#
# head_only/
#   config.json
#   diffusion_pytorch_model.safetensors
#
# partial/
#   config.json
#   diffusion_pytorch_model.safetensors
#
# etc.

CHECKPOINT_ROOT = Path(
    os.getenv(
        "DIFFUSION_CHECKPOINT_ROOT",
        "checkpoints/diffusion",
    )
)


DIFFUSION_CHECKPOINTS: dict[str, Optional[Path]] = {
    "zero_shot": None,

    "head_only": Path(
        os.getenv(
            "DIFFUSION_HEAD_ONLY_CHECKPOINT",
            str(CHECKPOINT_ROOT / "head_only"),
        )
    ),

    "partial": Path(
        os.getenv(
            "DIFFUSION_PARTIAL_CHECKPOINT",
            str(CHECKPOINT_ROOT / "partial"),
        )
    ),

    "lora": Path(
        os.getenv(
            "DIFFUSION_LORA_CHECKPOINT",
            str(CHECKPOINT_ROOT / "lora"),
        )
    ),

    "full": Path(
        os.getenv(
            "DIFFUSION_FULL_CHECKPOINT",
            str(CHECKPOINT_ROOT / "full"),
        )
    ),
}


SUPPORTED_STRATEGIES = (
    "zero_shot",
    "head_only",
    "partial",
    "lora",
    "full",
)


# =============================================================================
# Runtime state
# =============================================================================

_PIPELINE: Optional[StableDiffusionUpscalePipeline] = None
_PIPELINE_LOCK = threading.RLock()


# =============================================================================
# Device / dtype
# =============================================================================

def _device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16

    return torch.float32


# =============================================================================
# Checkpoint discovery
# =============================================================================

def get_diffusion_checkpoint(
    strategy: str,
) -> Optional[Path]:
    strategy = strategy.strip().lower()

    if strategy not in DIFFUSION_CHECKPOINTS:
        raise ValueError(
            f"Unknown diffusion strategy: {strategy}"
        )

    return DIFFUSION_CHECKPOINTS[strategy]


def _find_weights_file(
    checkpoint_dir: Path,
) -> Optional[Path]:
    """
    Find the weights file inside a Diffusers checkpoint directory.

    Standard Diffusers names are preferred, but the function also supports
    a directory containing exactly one .safetensors file.
    """

    preferred_names = (
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "pytorch_model.safetensors",
    )

    for name in preferred_names:
        candidate = checkpoint_dir / name

        if candidate.is_file():
            return candidate

    safetensors_files = sorted(
        checkpoint_dir.glob("*.safetensors")
    )

    if len(safetensors_files) == 1:
        return safetensors_files[0]

    return None


def _find_config_file(
    checkpoint_dir: Path,
) -> Optional[Path]:
    """
    Find the configuration file.

    Standard Diffusers UNet checkpoints use config.json.
    """

    preferred = checkpoint_dir / "config.json"

    if preferred.is_file():
        return preferred

    json_files = sorted(
        checkpoint_dir.glob("*.json")
    )

    if len(json_files) == 1:
        return json_files[0]

    return None


def _checkpoint_is_valid(
    checkpoint_dir: Optional[Path],
) -> bool:
    if checkpoint_dir is None:
        return False

    if not checkpoint_dir.is_dir():
        return False

    config_file = _find_config_file(checkpoint_dir)

    weights_file = _find_weights_file(checkpoint_dir)

    return (
        config_file is not None
        and weights_file is not None
    )


def is_diffusion_available(
    strategy: str,
) -> bool:
    strategy = strategy.strip().lower()

    if strategy == "zero_shot":
        return True

    checkpoint_dir = get_diffusion_checkpoint(
        strategy
    )

    return _checkpoint_is_valid(
        checkpoint_dir
    )


def diffusion_availability() -> dict[str, dict]:
    """
    Return runtime availability information used by the frontend.
    """

    result: dict[str, dict] = {}

    for strategy in SUPPORTED_STRATEGIES:

        checkpoint_dir = get_diffusion_checkpoint(
            strategy
        )

        if strategy == "zero_shot":

            result[strategy] = {
                "available": True,
                "checkpoint": DIFFUSION_MODEL_ID,
                "reason": None,
            }

            continue

        if _checkpoint_is_valid(
            checkpoint_dir
        ):

            weights_file = _find_weights_file(
                checkpoint_dir
            )

            config_file = _find_config_file(
                checkpoint_dir
            )

            result[strategy] = {
                "available": True,
                "checkpoint": str(
                    checkpoint_dir
                ),
                "weights": str(
                    weights_file
                ) if weights_file else None,
                "config": str(
                    config_file
                ) if config_file else None,
                "reason": None,
            }

        else:

            reason = "checkpoint directory not found"

            if checkpoint_dir is not None and checkpoint_dir.is_dir():

                config_file = _find_config_file(
                    checkpoint_dir
                )

                weights_file = _find_weights_file(
                    checkpoint_dir
                )

                if config_file is None:
                    reason = "config.json not found"

                elif weights_file is None:
                    reason = (
                        "safetensors weights not found"
                    )

            result[strategy] = {
                "available": False,
                "checkpoint": (
                    str(checkpoint_dir)
                    if checkpoint_dir
                    else None
                ),
                "reason": reason,
            }

    return result


# =============================================================================
# Base pipeline
# =============================================================================

def _make_pipeline(
) -> StableDiffusionUpscalePipeline:

    """
    Load the base Stable Diffusion x4 Upscaler.

    The downloaded local model is used through Diffusers. We first try
    local_files_only=True because the project keeps the pretrained model
    locally.
    """

    try:

        pipe = (
            StableDiffusionUpscalePipeline.from_pretrained(
                DIFFUSION_MODEL_ID,
                torch_dtype=_dtype(),
                safety_checker=None,
                local_files_only=True,
            )
        )

    except Exception as local_exc:

        # This fallback allows the model to be downloaded if it is not yet
        # cached locally.
        print(
            "[Diffusion] local model loading failed:"
        )

        print(
            f"[Diffusion] {local_exc}"
        )

        print(
            "[Diffusion] trying normal from_pretrained loading..."
        )

        pipe = (
            StableDiffusionUpscalePipeline.from_pretrained(
                DIFFUSION_MODEL_ID,
                torch_dtype=_dtype(),
                safety_checker=None,
            )
        )

    if torch.cuda.is_available():

        if DIFFUSION_CPU_OFFLOAD:

            pipe.enable_model_cpu_offload()

        else:

            pipe = pipe.to("cuda")

    pipe.set_progress_bar_config(
        disable=True
    )

    # Mark the initial pipeline as using the base checkpoint.
    setattr(
        pipe,
        "_sr_loaded_strategy",
        "zero_shot",
    )

    return pipe


def get_diffusion_pipeline(
) -> StableDiffusionUpscalePipeline:

    global _PIPELINE

    if _PIPELINE is None:

        with _PIPELINE_LOCK:

            if _PIPELINE is None:

                _PIPELINE = _make_pipeline()

    return _PIPELINE


# =============================================================================
# Checkpoint config inspection
# =============================================================================

def _read_checkpoint_config(
    checkpoint_dir: Path,
) -> dict:
    """
    Read the local config.json.

    We do not use the config to reconstruct the entire pipeline because the
    resident pipeline already contains the correct Stable Diffusion x4 UNet.
    Reading it here gives us a useful compatibility diagnostic.
    """

    config_path = _find_config_file(
        checkpoint_dir
    )

    if config_path is None:
        raise FileNotFoundError(
            f"Could not find config.json in "
            f"{checkpoint_dir}"
        )

    try:

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            f"Invalid JSON configuration: "
            f"{config_path}"
        ) from exc


# =============================================================================
# State-dict key normalization
# =============================================================================

def _normalise_state_dict_keys(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:

    """
    Remove common training wrappers.

    This allows checkpoints saved with prefixes such as:

        module.xxx
        unet.xxx
        base_model.model.xxx

    to be loaded into the pipeline's UNet.
    """

    prefixes = (
        "module.",
        "unet.",
        "base_model.model.",
    )

    result: dict[str, torch.Tensor] = {}

    for key, value in state.items():

        new_key = key

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(prefix):

                    new_key = new_key[
                        len(prefix):
                    ]

                    changed = True

                    break

        result[new_key] = value

    return result


# =============================================================================
# SafeTensors inspection
# =============================================================================

def _load_safetensors_state(
    weights_file: Path,
) -> dict[str, torch.Tensor]:

    if not weights_file.is_file():

        raise FileNotFoundError(
            f"Diffusion weights not found: "
            f"{weights_file}"
        )

    print(
        f"[Diffusion] reading SafeTensors: "
        f"{weights_file}"
    )

    state = load_safetensors(
        str(weights_file),
        device="cpu",
    )

    if not state:

        raise RuntimeError(
            f"SafeTensors checkpoint is empty: "
            f"{weights_file}"
        )

    return _normalise_state_dict_keys(
        state
    )


# =============================================================================
# Fine-tuned UNet loading
# =============================================================================

def _load_unet_checkpoint(
    pipe: StableDiffusionUpscalePipeline,
    checkpoint_dir: Path,
) -> None:

    """
    Load a Diffusers-format fine-tuned UNet.

    Expected directory:

        checkpoint_dir/
            config.json
            *.safetensors

    The weights are loaded directly into the resident pipeline UNet.
    """

    if not checkpoint_dir.is_dir():

        raise FileNotFoundError(
            "Diffusion checkpoint directory "
            f"not found: {checkpoint_dir}"
        )

    config = _read_checkpoint_config(
        checkpoint_dir
    )

    weights_file = _find_weights_file(
        checkpoint_dir
    )

    if weights_file is None:

        raise FileNotFoundError(
            "Could not find a .safetensors "
            f"weights file in {checkpoint_dir}"
        )

    print()
    print(
        "=================================================="
    )
    print(
        "[Diffusion] Loading fine-tuned checkpoint"
    )
    print(
        f"[Diffusion] directory : {checkpoint_dir}"
    )
    print(
        f"[Diffusion] weights   : {weights_file.name}"
    )
    print(
        f"[Diffusion] config    : "
        f"{_find_config_file(checkpoint_dir).name}"
    )
    print(
        "=================================================="
    )

    # -------------------------------------------------------------------------
    # Compare the checkpoint configuration with the resident UNet where
    # possible. We don't fail merely because a config contains extra fields.
    # -------------------------------------------------------------------------

    try:

        base_config = pipe.unet.config

        checkpoint_sample_size = config.get(
            "sample_size"
        )

        base_sample_size = getattr(
            base_config,
            "sample_size",
            None,
        )

        if (
            checkpoint_sample_size is not None
            and base_sample_size is not None
            and checkpoint_sample_size
            != base_sample_size
        ):

            print(
                "[Diffusion] warning: checkpoint "
                "sample_size differs from base UNet"
            )

        checkpoint_in_channels = config.get(
            "in_channels"
        )

        base_in_channels = getattr(
            base_config,
            "in_channels",
            None,
        )

        if (
            checkpoint_in_channels is not None
            and base_in_channels is not None
            and checkpoint_in_channels
            != base_in_channels
        ):

            raise RuntimeError(
                "The fine-tuned checkpoint has "
                f"in_channels={checkpoint_in_channels}, "
                "but the loaded Stable Diffusion "
                f"x4 Upscaler UNet expects "
                f"in_channels={base_in_channels}."
            )

        checkpoint_out_channels = config.get(
            "out_channels"
        )

        base_out_channels = getattr(
            base_config,
            "out_channels",
            None,
        )

        if (
            checkpoint_out_channels is not None
            and base_out_channels is not None
            and checkpoint_out_channels
            != base_out_channels
        ):

            raise RuntimeError(
                "The fine-tuned checkpoint has "
                f"out_channels={checkpoint_out_channels}, "
                "but the loaded Stable Diffusion "
                f"x4 Upscaler UNet expects "
                f"out_channels={base_out_channels}."
            )

    except AttributeError:
        pass

    # -------------------------------------------------------------------------
    # Load SafeTensors into CPU memory.
    # -------------------------------------------------------------------------

    state = _load_safetensors_state(
        weights_file
    )

    print(
        f"[Diffusion] checkpoint tensors: "
        f"{len(state)}"
    )

    # -------------------------------------------------------------------------
    # Compare keys before modifying the resident UNet.
    # -------------------------------------------------------------------------

    model_state = pipe.unet.state_dict()

    model_keys = set(
        model_state.keys()
    )

    checkpoint_keys = set(
        state.keys()
    )

    matching_keys = (
        model_keys & checkpoint_keys
    )

    missing_keys = (
        model_keys - checkpoint_keys
    )

    unexpected_keys = (
        checkpoint_keys - model_keys
    )

    print(
        f"[Diffusion] matching tensors : "
        f"{len(matching_keys)}"
    )

    print(
        f"[Diffusion] missing tensors  : "
        f"{len(missing_keys)}"
    )

    print(
        f"[Diffusion] unexpected       : "
        f"{len(unexpected_keys)}"
    )

    # -------------------------------------------------------------------------
    # We require at least one matching parameter.
    # -------------------------------------------------------------------------

    if len(matching_keys) == 0:

        sample_checkpoint = list(
            checkpoint_keys
        )[:10]

        sample_model = list(
            model_keys
        )[:10]

        raise RuntimeError(
            "The fine-tuned SafeTensors file does not "
            "contain keys matching the resident UNet.\n\n"
            f"Checkpoint sample keys:\n"
            f"{sample_checkpoint}\n\n"
            f"Model sample keys:\n"
            f"{sample_model}"
        )

    # -------------------------------------------------------------------------
    # Detect whether this is a complete UNet checkpoint or a partial one.
    # -------------------------------------------------------------------------

    if (
        len(missing_keys) == 0
        and len(unexpected_keys) == 0
    ):

        load_mode = "complete"

    else:

        load_mode = "partial"

    print(
        f"[Diffusion] checkpoint type: {load_mode}"
    )

    # -------------------------------------------------------------------------
    # Load.
    #
    # strict=False is deliberate.
    #
    # head_only / partial checkpoints can contain only the tensors that were
    # fine-tuned. The untouched base UNet parameters should remain exactly as
    # provided by the original Stable Diffusion x4 Upscaler.
    # -------------------------------------------------------------------------

    missing, unexpected = pipe.unet.load_state_dict(
        state,
        strict=False,
    )

    if unexpected:

        print(
            "[Diffusion] warning: "
            f"{len(unexpected)} unexpected keys"
        )

    if missing:

        print(
            "[Diffusion] "
            f"{len(missing)} base UNet keys were "
            "left unchanged"
        )

    # -------------------------------------------------------------------------
    # Clean CPU checkpoint memory immediately.
    # -------------------------------------------------------------------------

    del state

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print(
        "[Diffusion] fine-tuned UNet loaded successfully."
    )

    print(
        "=================================================="
    )
    print()


# =============================================================================
# Strategy preparation
# =============================================================================

def prepare_diffusion_strategy(
    strategy: str,
) -> StableDiffusionUpscalePipeline:

    """
    Prepare the requested strategy.

    zero_shot:
        original pretrained UNet

    head_only / partial / lora / full:
        local fine-tuned UNet checkpoint

    IMPORTANT:
        All strategies reuse one resident pipeline to avoid keeping multiple
        4.7e8-parameter diffusion pipelines in memory simultaneously.
    """

    strategy = strategy.strip().lower()

    if strategy not in SUPPORTED_STRATEGIES:

        raise ValueError(
            f"Unknown diffusion strategy: "
            f"{strategy}"
        )

    pipe = get_diffusion_pipeline()

    # -------------------------------------------------------------------------
    # Zero-shot
    # -------------------------------------------------------------------------

    if strategy == "zero_shot":

        # Once an adapted checkpoint has replaced the resident UNet, we do not
        # silently call the adapted model "zero-shot".
        #
        # Restarting the worker restores the original pretrained pipeline.

        loaded_strategy = getattr(
            pipe,
            "_sr_loaded_strategy",
            "zero_shot",
        )

        if loaded_strategy != "zero_shot":

            raise RuntimeError(
                "The diffusion worker currently has "
                f"the '{loaded_strategy}' fine-tuned UNet loaded. "
                "Restart the backend before running zero-shot."
            )

        return pipe

    # -------------------------------------------------------------------------
    # Fine-tuned strategy
    # -------------------------------------------------------------------------

    checkpoint_dir = get_diffusion_checkpoint(
        strategy
    )

    if (
        checkpoint_dir is None
        or not _checkpoint_is_valid(
            checkpoint_dir
        )
    ):

        raise FileNotFoundError(
            f"Diffusion strategy '{strategy}' "
            "is unavailable.\n"
            f"Expected checkpoint directory: "
            f"{checkpoint_dir}"
        )

    loaded_strategy = getattr(
        pipe,
        "_sr_loaded_strategy",
        None,
    )

    # Don't reload the same checkpoint for every request.
    if loaded_strategy != strategy:

        _load_unet_checkpoint(
            pipe,
            checkpoint_dir,
        )

        setattr(
            pipe,
            "_sr_loaded_strategy",
            strategy,
        )

    return pipe


# =============================================================================
# Image utilities
# =============================================================================

def _tensor_to_pil(
    x: torch.Tensor,
) -> Image.Image:

    x = (
        x.detach()
        .cpu()
        .float()
        .clamp(0, 1)
    )

    arr = (
        x.numpy()
        .transpose(1, 2, 0)
        * 255.0
    ).round().astype(np.uint8)

    return Image.fromarray(
        arr,
        mode="RGB",
    )


def _resize_exact(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:

    if image.size == (
        width,
        height,
    ):

        return image

    array = (
        np.asarray(image)
        .astype(np.float32)
        / 255.0
    )

    tensor = (
        torch.from_numpy(
            array.transpose(2, 0, 1)
        )
        .unsqueeze(0)
    )

    tensor = F.interpolate(
        tensor,
        size=(
            height,
            width,
        ),
        mode="bicubic",
        align_corners=False,
    )

    result = (
        tensor[0]
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )

    return Image.fromarray(
        result,
        mode="RGB",
    )


# =============================================================================
# Main inference function
# =============================================================================

def run_diffusion(
    lr_image: Image.Image,
    strategy: str = "zero_shot",
    seed: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
) -> Image.Image:

    """
    Run 4x diffusion super-resolution.

    Parameters
    ----------
    lr_image:
        Input low-resolution RGB image.

    strategy:
        zero_shot
        head_only
        partial
        lora
        full

    seed:
        Random seed used by the diffusion generator.

    num_inference_steps:
        Number of DDIM inference steps.

    Returns
    -------
    PIL.Image.Image
        4x super-resolved RGB image.
    """

    strategy = strategy.strip().lower()

    if strategy not in SUPPORTED_STRATEGIES:

        raise ValueError(
            f"Unknown diffusion strategy: "
            f"{strategy}"
        )

    # -------------------------------------------------------------------------
    # Availability check
    # -------------------------------------------------------------------------

    if not is_diffusion_available(
        strategy
    ):

        checkpoint_dir = (
            get_diffusion_checkpoint(
                strategy
            )
        )

        raise FileNotFoundError(
            f"Diffusion strategy '{strategy}' "
            "is unavailable.\n"
            f"Checkpoint: {checkpoint_dir}"
        )

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    lr_image = lr_image.convert(
        "RGB"
    )

    width, height = lr_image.size

    if width < 1 or height < 1:

        raise ValueError(
            "Input image has an invalid size."
        )

    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------

    seed = (
        DIFFUSION_SEED_BASE
        if seed is None
        else int(seed)
    )

    # -------------------------------------------------------------------------
    # Sampling steps
    # -------------------------------------------------------------------------

    steps = (
        DIFFUSION_STEPS
        if num_inference_steps is None
        else int(num_inference_steps)
    )

    if steps < 1 or steps > 100:

        raise ValueError(
            "num_inference_steps must be "
            "between 1 and 100."
        )

    # -------------------------------------------------------------------------
    # Serialize pipeline access.
    #
    # This is especially important because switching strategies modifies
    # pipe.unet.
    # -------------------------------------------------------------------------

    with _PIPELINE_LOCK:

        pipe = prepare_diffusion_strategy(
            strategy
        )

        generator_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        generator = (
            torch.Generator(
                device=generator_device
            )
            .manual_seed(seed)
        )

        print(
            f"[Diffusion] inference strategy={strategy} "
            f"steps={steps} "
            f"seed={seed}"
        )

        with torch.inference_mode():

            result = pipe(
                prompt=DIFFUSION_PROMPT,
                image=lr_image,
                noise_level=DIFFUSION_NOISE_LEVEL,
                guidance_scale=DIFFUSION_GUIDANCE_SCALE,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

    # -------------------------------------------------------------------------
    # Exact 4x output guard
    # -------------------------------------------------------------------------

    result = _resize_exact(
        result,
        width * 4,
        height * 4,
    )

    return result


# =============================================================================
# PNG response helper
# =============================================================================

def image_to_png_bytes(
    image: Image.Image,
) -> bytes:

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# =============================================================================
# Runtime information
# =============================================================================

def diffusion_runtime_info() -> dict:
    """
    Return information useful to the FastAPI endpoint/frontend.
    """

    device = _device()

    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dtype": str(_dtype()),
        "model_id": DIFFUSION_MODEL_ID,
        "steps": DIFFUSION_STEPS,
        "noise_level": DIFFUSION_NOISE_LEVEL,
        "guidance_scale": DIFFUSION_GUIDANCE_SCALE,
        "prompt": DIFFUSION_PROMPT,
        "seed": DIFFUSION_SEED_BASE,
        "cpu_offload": DIFFUSION_CPU_OFFLOAD,
        "checkpoint_root": str(
            CHECKPOINT_ROOT
        ),
        "strategies": diffusion_availability(),
    }